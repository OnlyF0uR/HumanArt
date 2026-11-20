//! The DEFLATE encoder `[IO-01]`: hash-chain LZ77 with lazy matching, and per block the cheapest of
//! a dynamic Huffman code, the fixed code and stored bytes.
//!
//! The match finder keeps, for every three-byte hash, the chain of earlier positions with that hash
//! (as zlib does). Positions are stored modulo 2^16; a stale or aliased entry is harmless because
//! every candidate is verified against the input before it is used. Code lengths are limited to 15
//! bits (7 for the code-length code) by building the optimal code and then moving the too-deep
//! codewords up, which keeps the code complete.

use super::tables::{
    DIST_BASE, DIST_EXTRA, END_OF_BLOCK, FIXED_DIST_LEN, LEN_BASE, LEN_EXTRA, MAX_MATCH, MIN_MATCH, NUM_DIST,
    NUM_LITLEN, NUM_PRECODE, PRECODE_ORDER, WINDOW, fixed_litlen_len,
};

/// How hard the encoder works for a smaller output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Level {
    /// No compression: stored blocks only.
    Store,
    /// Greedy matching over short hash chains.
    Fast,
    /// Lazy matching over moderate chains (comparable to zlib's level 6).
    #[default]
    Default,
    /// Lazy matching over long chains (comparable to zlib's level 9).
    Best,
}

/// The match finder's effort for one [`Level`].
struct Params {
    /// How many earlier positions are tried per match.
    max_chain: u32,
    /// A match this long ends the search.
    nice: usize,
    /// With a pending match this long, the next search tries a quarter of the chain.
    good: usize,
    /// A match shorter than this waits one position for a longer one (0: never wait).
    max_lazy: usize,
    /// The positions inside a match longer than this are not entered in the hash chains.
    max_insert: usize,
}

impl Level {
    const fn params(self) -> Params {
        match self {
            Self::Store | Self::Fast => Params { max_chain: 8, nice: 32, good: 8, max_lazy: 0, max_insert: 32 },
            Self::Default => Params { max_chain: 128, nice: 128, good: 8, max_lazy: 16, max_insert: MAX_MATCH },
            Self::Best => {
                Params { max_chain: 4096, nice: MAX_MATCH, good: 32, max_lazy: MAX_MATCH, max_insert: MAX_MATCH }
            }
        }
    }
}

const HASH_BITS: u32 = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
const WINDOW_MASK: usize = WINDOW - 1;
/// Symbols (literals and matches) per block.
const BLOCK_TOKENS: usize = 16_384;
/// A match of the minimum length further back than this costs more than three literals.
const TOO_FAR: usize = 4096;
/// Most bytes a stored block holds.
const STORED_MAX: usize = 65_535;
/// Symbols of the fixed literal/length code.
const FIXED_LITLEN: usize = 288;
/// Longest codeword of the literal/length and distance codes.
const MAX_CODE_LEN: usize = 15;
/// Longest codeword of the code-length code.
const MAX_PRECODE_LEN: usize = 7;

/// Length symbol (0..=28, to be added to 257) of match length `index + 3`.
const LEN_SYM: [u8; 256] = make_len_sym();
/// Distance symbol of distance `d + 1` for `d < 256`, and of `256 + (d >> 7)` beyond.
const DIST_SYM: [u8; 512] = make_dist_sym();

const fn make_len_sym() -> [u8; 256] {
    let mut table = [0_u8; 256];
    let mut sym = 0;
    while sym < LEN_BASE.len() {
        let first = LEN_BASE[sym] as usize - MIN_MATCH;
        let count = 1_usize << LEN_EXTRA[sym];
        let mut i = 0;
        // Length 258 has its own symbol, which comes last and replaces the entry of symbol 284.
        while i < count && first + i < table.len() {
            table[first + i] = sym as u8;
            i += 1;
        }
        sym += 1;
    }
    table
}

const fn make_dist_sym() -> [u8; 512] {
    let mut table = [0_u8; 512];
    let mut sym = 0;
    while sym < DIST_BASE.len() {
        let first = DIST_BASE[sym] as usize - 1;
        let count = 1_usize << DIST_EXTRA[sym];
        let mut i = 0;
        while i < count {
            let d = first + i;
            if d < 256 {
                table[d] = sym as u8;
                i += 1;
            } else {
                // From distance 257 on, every symbol covers whole multiples of 128.
                table[256 + (d >> 7)] = sym as u8;
                i += 128;
            }
        }
        sym += 1;
    }
    table
}

/// The distance symbol of `distance` (1..=32768).
#[inline]
fn dist_symbol(distance: usize) -> usize {
    let d = distance - 1;
    usize::from(if d < 256 { DIST_SYM[d] } else { DIST_SYM[(256 + (d >> 7)) & 511] })
}

/// Collects bits, lowest first, into bytes.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    buf: u64,
    cnt: u32,
}

impl BitWriter<'_> {
    /// Appends the low `n` bits of `value` (`n` at most 32, the bits above them zero).
    #[inline]
    fn put(&mut self, value: u32, n: u32) {
        self.buf |= u64::from(value) << self.cnt;
        self.cnt += n;
        if self.cnt >= 32 {
            self.out.extend_from_slice(&(self.buf as u32).to_le_bytes());
            self.buf >>= 32;
            self.cnt -= 32;
        }
    }

    /// Pads with zero bits to the next byte boundary and writes everything out.
    fn align(&mut self) {
        while self.cnt > 0 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.cnt = self.cnt.saturating_sub(8);
        }
    }
}

/// Computes codeword lengths of at most `limit` bits for symbols with the given frequencies.
///
/// Unused symbols get length 0, except that at least two symbols always get a codeword, so the code
/// is complete and no decoder has to deal with a one-symbol code. `lens` must be as long as `freqs`
/// (2 to 288 symbols) and `limit` at most 15.
fn build_lengths(freqs: &[u32], limit: usize, lens: &mut [u8]) {
    const MAX: usize = FIXED_LITLEN;
    lens.fill(0);
    let mut leaves = [(0_u32, 0_u16); MAX];
    let mut m = 0;
    for (sym, &freq) in freqs.iter().enumerate().take(MAX.min(lens.len())) {
        if freq != 0 {
            leaves[m] = (freq, sym as u16);
            m += 1;
        }
    }
    if m < 2 {
        let only = if m == 1 { usize::from(leaves[0].1) } else { 0 };
        for sym in [only, usize::from(only == 0)] {
            if let Some(len) = lens.get_mut(sym) {
                *len = 1;
            }
        }
        return;
    }
    let leaves = &mut leaves[..m];
    leaves.sort_unstable();

    // Huffman's algorithm with two queues: the sorted leaves, and the internal nodes, which are
    // made in non-decreasing order of weight. Node ids: leaves 0..m, internal nodes m..2m-1.
    let mut weight = [0_u64; MAX];
    let mut parent = [0_u16; 2 * MAX];
    let (mut next_leaf, mut next_inner, mut made) = (0, 0, 0);
    while made < m - 1 {
        let mut sum = 0;
        for _ in 0..2 {
            let leaf_is_lighter =
                next_leaf < m && (next_inner >= made || u64::from(leaves[next_leaf].0) <= weight[next_inner]);
            if leaf_is_lighter {
                sum += u64::from(leaves[next_leaf].0);
                parent[next_leaf] = (m + made) as u16;
                next_leaf += 1;
            } else {
                sum += weight[next_inner];
                parent[m + next_inner] = (m + made) as u16;
                next_inner += 1;
            }
        }
        weight[made] = sum;
        made += 1;
    }
    // Depths from the root (the last internal node) down.
    let mut depth = [0_u16; MAX];
    for inner in (0..m - 2).rev() {
        depth[inner] = depth[usize::from(parent[m + inner]) - m] + 1;
    }
    let mut count = [0_u32; MAX_CODE_LEN + 1];
    for &up in &parent[..m] {
        let len = usize::from(depth[usize::from(up) - m] + 1).min(limit);
        count[len] += 1;
    }
    // Clamping the deep leaves to `limit` over-fills the code space; in units of the longest
    // codeword, give back one unit at a time by moving a shorter codeword one level down next to
    // one taken from the deepest level.
    let mut total: u64 = (1..=limit).map(|len| u64::from(count[len]) << (limit - len)).sum();
    while total > 1 << limit {
        count[limit] = count[limit].saturating_sub(1);
        for len in (1..limit).rev() {
            if count[len] > 0 {
                count[len] -= 1;
                count[len + 1] += 2;
                break;
            }
        }
        total -= 1;
    }
    // The rarest symbols get the longest codewords.
    let mut next = 0;
    for len in (1..=limit).rev() {
        for _ in 0..count[len] {
            if let Some(&(_, sym)) = leaves.get(next) {
                lens[usize::from(sym)] = len as u8;
            }
            next += 1;
        }
    }
}

/// Assigns the canonical codewords for `lens`, bit-reversed so that they can be written lowest bit
/// first.
fn make_codes(lens: &[u8], codes: &mut [u16]) {
    let mut count = [0_u32; MAX_CODE_LEN + 1];
    for &len in lens {
        count[usize::from(len) & MAX_CODE_LEN] += 1;
    }
    count[0] = 0;
    let mut next = [0_u32; MAX_CODE_LEN + 1];
    let mut code = 0_u32;
    for len in 1..=MAX_CODE_LEN {
        code = (code + count[len - 1]) << 1;
        next[len] = code;
    }
    for (&len, slot) in lens.iter().zip(codes.iter_mut()) {
        if len != 0 {
            let len = usize::from(len) & MAX_CODE_LEN;
            *slot = ((next[len] as u16).reverse_bits()) >> (16 - len);
            next[len] += 1;
        }
    }
}

/// The symbols of the block being collected, and how often each occurs.
struct Block {
    /// A literal is its byte; a match is `length << 16 | distance`.
    tokens: Vec<u32>,
    lit_freq: [u32; NUM_LITLEN],
    dist_freq: [u32; NUM_DIST],
}

impl Block {
    fn new() -> Self {
        Self { tokens: Vec::with_capacity(BLOCK_TOKENS), lit_freq: [0; NUM_LITLEN], dist_freq: [0; NUM_DIST] }
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.lit_freq = [0; NUM_LITLEN];
        self.dist_freq = [0; NUM_DIST];
    }

    #[inline]
    fn literal(&mut self, byte: u8) {
        self.tokens.push(u32::from(byte));
        self.lit_freq[usize::from(byte)] += 1;
    }

    /// A match of `len` (3..=258) bytes, `distance` (1..=32768) back.
    #[inline]
    fn matched(&mut self, len: usize, distance: usize) {
        self.tokens.push(((len as u32) << 16) | distance as u32);
        self.lit_freq[257 + usize::from(LEN_SYM[(len - MIN_MATCH) & 255])] += 1;
        self.dist_freq[dist_symbol(distance)] += 1;
    }
}

/// One Huffman code: codeword lengths and bit-reversed codewords.
struct Code<const N: usize> {
    lens: [u8; N],
    codes: [u16; N],
}

impl<const N: usize> Code<N> {
    fn from_freqs(freqs: &[u32; N], limit: usize) -> Self {
        let mut code = Self { lens: [0; N], codes: [0; N] };
        build_lengths(freqs, limit, &mut code.lens);
        make_codes(&code.lens, &mut code.codes);
        code
    }

    fn from_lens(lens: [u8; N]) -> Self {
        let mut code = Self { lens, codes: [0; N] };
        make_codes(&code.lens, &mut code.codes);
        code
    }

    #[inline]
    fn put(&self, w: &mut BitWriter<'_>, sym: usize) {
        if let (Some(&code), Some(&len)) = (self.codes.get(sym), self.lens.get(sym)) {
            w.put(u32::from(code), u32::from(len));
        }
    }

    /// The number of symbols up to the last one with a codeword, at least `min`.
    fn used(&self, min: usize) -> usize {
        self.lens.iter().rposition(|&len| len != 0).map_or(min, |last| (last + 1).max(min))
    }
}

/// Writes the tokens and the end-of-block symbol with the given codes.
fn write_tokens<const L: usize, const D: usize>(
    w: &mut BitWriter<'_>,
    tokens: &[u32],
    litlen: &Code<L>,
    dist: &Code<D>,
) {
    for &token in tokens {
        let len = (token >> 16) as usize;
        if len == 0 {
            litlen.put(w, (token & 0xFF) as usize);
            continue;
        }
        let len_sym = usize::from(LEN_SYM[(len - MIN_MATCH) & 255]);
        litlen.put(w, 257 + len_sym);
        let extra = u32::from(LEN_EXTRA[len_sym]);
        if extra != 0 {
            w.put((len - usize::from(LEN_BASE[len_sym])) as u32, extra);
        }
        let distance = (token & 0xFFFF) as usize;
        let dist_sym = dist_symbol(distance);
        dist.put(w, dist_sym);
        let extra = u32::from(DIST_EXTRA[dist_sym]);
        if extra != 0 {
            w.put((distance - usize::from(DIST_BASE[dist_sym])) as u32, extra);
        }
    }
    litlen.put(w, END_OF_BLOCK);
}

/// The bits the tokens take with the given codeword lengths (the end-of-block symbol included, which
/// `lit_freq` counts).
fn token_bits(lit_freq: &[u32; NUM_LITLEN], dist_freq: &[u32; NUM_DIST], lit_lens: &[u8], dist_lens: &[u8]) -> u64 {
    let mut bits = 0_u64;
    for (sym, (&freq, &len)) in lit_freq.iter().zip(lit_lens).enumerate() {
        let extra = if sym > END_OF_BLOCK { LEN_EXTRA[sym - 257] } else { 0 };
        bits += u64::from(freq) * u64::from(len + extra);
    }
    for (sym, (&freq, &len)) in dist_freq.iter().zip(dist_lens).enumerate() {
        bits += u64::from(freq) * u64::from(len + DIST_EXTRA[sym]);
    }
    bits
}

/// Writes `raw` as stored blocks (at least one, even when `raw` is empty).
fn write_stored(w: &mut BitWriter<'_>, raw: &[u8], last: bool) {
    let mut rest = raw;
    loop {
        let (chunk, tail) = rest.split_at(rest.len().min(STORED_MAX));
        rest = tail;
        w.put(u32::from(last && rest.is_empty()), 1);
        w.put(0, 2);
        w.align();
        let len = chunk.len() as u16;
        w.out.extend_from_slice(&len.to_le_bytes());
        w.out.extend_from_slice(&(!len).to_le_bytes());
        w.out.extend_from_slice(chunk);
        if rest.is_empty() {
            return;
        }
    }
}

/// The code lengths of a dynamic block run-length coded with symbols 16, 17 and 18, as
/// `(symbol, extra value)` pairs.
fn run_length_code(lens: &[u8], out: &mut [(u8, u8); NUM_LITLEN + NUM_DIST]) -> usize {
    let mut n = 0;
    let mut push = |sym: u8, extra: usize| {
        if let Some(slot) = out.get_mut(n) {
            *slot = (sym, extra as u8);
            n += 1;
        }
    };
    let mut i = 0;
    while i < lens.len() {
        let value = lens[i];
        let mut run = 1;
        while i + run < lens.len() && lens[i + run] == value {
            run += 1;
        }
        i += run;
        if value == 0 {
            while run >= 11 {
                let step = run.min(138);
                push(18, step - 11);
                run -= step;
            }
            if run >= 3 {
                push(17, run - 3);
                run = 0;
            }
        } else {
            push(value, 0);
            run -= 1;
            while run >= 3 {
                let step = run.min(6);
                push(16, step - 3);
                run -= step;
            }
        }
        for _ in 0..run {
            push(value, 0);
        }
    }
    n
}

/// Extra bits of the code-length code's symbols 16, 17 and 18.
const fn precode_extra(sym: u8) -> u32 {
    match sym {
        16 => 2,
        17 => 3,
        18 => 7,
        _ => 0,
    }
}

/// Writes one block holding `block`'s tokens, which encode `raw`, in the cheapest of the three block
/// types.
fn write_block(w: &mut BitWriter<'_>, block: &mut Block, raw: &[u8], last: bool) {
    block.lit_freq[END_OF_BLOCK] = 1;
    let litlen = Code::<NUM_LITLEN>::from_freqs(&block.lit_freq, MAX_CODE_LEN);
    let dist = Code::<NUM_DIST>::from_freqs(&block.dist_freq, MAX_CODE_LEN);
    let num_litlen = litlen.used(257);
    let num_dist = dist.used(1);

    let mut all_lens = [0_u8; NUM_LITLEN + NUM_DIST];
    all_lens[..num_litlen].copy_from_slice(&litlen.lens[..num_litlen]);
    all_lens[num_litlen..num_litlen + num_dist].copy_from_slice(&dist.lens[..num_dist]);
    let mut runs = [(0_u8, 0_u8); NUM_LITLEN + NUM_DIST];
    let num_runs = run_length_code(&all_lens[..num_litlen + num_dist], &mut runs);
    let runs = &runs[..num_runs];
    let mut precode_freq = [0_u32; NUM_PRECODE];
    for &(sym, _) in runs {
        precode_freq[usize::from(sym) % NUM_PRECODE] += 1;
    }
    let precode = Code::<NUM_PRECODE>::from_freqs(&precode_freq, MAX_PRECODE_LEN);
    let num_precode =
        PRECODE_ORDER.iter().rposition(|&sym| precode.lens[usize::from(sym)] != 0).map_or(4, |last| (last + 1).max(4));

    let header_bits: u64 = 14
        + 3 * num_precode as u64
        + runs
            .iter()
            .map(|&(sym, _)| u64::from(precode.lens[usize::from(sym) % NUM_PRECODE]) + u64::from(precode_extra(sym)))
            .sum::<u64>();
    let dynamic_bits = 3 + header_bits + token_bits(&block.lit_freq, &block.dist_freq, &litlen.lens, &dist.lens);
    let mut fixed_lens = [0_u8; FIXED_LITLEN];
    for (sym, len) in fixed_lens.iter_mut().enumerate() {
        *len = fixed_litlen_len(sym);
    }
    let fixed_bits = 3 + token_bits(&block.lit_freq, &block.dist_freq, &fixed_lens, &[FIXED_DIST_LEN; NUM_DIST]);
    // Per stored block: the 3 header bits, the padding to the byte boundary, LEN and its complement.
    let padding = u64::from((8 - (w.cnt + 3) % 8) % 8);
    let stored_bits = raw.len() as u64 * 8 + 3 + padding + 32 + 40 * (raw.len().div_ceil(STORED_MAX).max(1) as u64 - 1);

    if stored_bits <= dynamic_bits.min(fixed_bits) {
        write_stored(w, raw, last);
    } else if fixed_bits <= dynamic_bits {
        w.put(u32::from(last), 1);
        w.put(1, 2);
        let fixed_dist = Code::<32>::from_lens([FIXED_DIST_LEN; 32]);
        write_tokens(w, &block.tokens, &Code::<FIXED_LITLEN>::from_lens(fixed_lens), &fixed_dist);
    } else {
        w.put(u32::from(last), 1);
        w.put(2, 2);
        w.put((num_litlen - 257) as u32, 5);
        w.put((num_dist - 1) as u32, 5);
        w.put((num_precode - 4) as u32, 4);
        for &sym in PRECODE_ORDER.iter().take(num_precode) {
            w.put(u32::from(precode.lens[usize::from(sym)]), 3);
        }
        for &(sym, extra) in runs {
            precode.put(w, usize::from(sym));
            let extra_bits = precode_extra(sym);
            if extra_bits != 0 {
                w.put(u32::from(extra), extra_bits);
            }
        }
        write_tokens(w, &block.tokens, &litlen, &dist);
    }
    block.clear();
}

/// The hash of the three bytes at `at` (which must exist).
#[inline]
fn hash3(input: &[u8], at: usize) -> usize {
    let v = u32::from(input[at]) | (u32::from(input[at + 1]) << 8) | (u32::from(input[at + 2]) << 16);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

/// How many leading bytes of `a` and `b` are equal, up to `max` (both hold at least `max` bytes).
#[inline]
fn match_len(a: &[u8], b: &[u8], max: usize) -> usize {
    let (a, b) = (&a[..max], &b[..max]);
    let mut i = 0;
    for (x, y) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0) {
        let (x, y) = (u64::from_le_bytes(*x), u64::from_le_bytes(*y));
        if x != y {
            return i + ((x ^ y).trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < max && a[i] == b[i] {
        i += 1;
    }
    i
}

/// The hash chains: for every hash the latest position with it, and for every position of the window
/// the one before it with the same hash. Positions are kept modulo 2^16.
struct Chains {
    head: Vec<u16>,
    prev: Vec<u16>,
}

impl Chains {
    fn new() -> Self {
        Self { head: vec![0; HASH_SIZE], prev: vec![0; WINDOW] }
    }

    /// Enters position `at` (three bytes must follow it) and returns the previous head of its chain.
    #[inline]
    fn insert(&mut self, input: &[u8], at: usize) -> u16 {
        let hash = hash3(input, at);
        let before = self.head[hash];
        self.prev[at & WINDOW_MASK] = before;
        self.head[hash] = at as u16;
        before
    }

    /// Enters the positions `from..to` that have three bytes after them.
    fn insert_range(&mut self, input: &[u8], from: usize, to: usize) {
        for at in from..to.min(input.len().saturating_sub(MIN_MATCH - 1)) {
            self.insert(input, at);
        }
    }

    /// The longest match for `pos` that is longer than `floor`, as `(length, distance)`, or `(0, 0)`.
    /// `candidate` is the head of the chain for `pos`, read before `pos` was entered.
    fn longest_match(
        &self,
        input: &[u8],
        pos: usize,
        mut candidate: u16,
        max_chain: u32,
        floor: usize,
        nice: usize,
    ) -> (usize, usize) {
        let max_len = (input.len() - pos).min(MAX_MATCH);
        let mut best_len = floor;
        let mut best_distance = 0;
        if best_len >= max_len {
            return (0, 0);
        }
        let nice = nice.min(max_len);
        let mut chain = max_chain.max(1);
        let mut last_distance = 0;
        loop {
            let distance = usize::from((pos as u16).wrapping_sub(candidate));
            // A real chain goes strictly further back at every step; anything else is a stale entry.
            if distance == 0 || distance > WINDOW || distance > pos || distance <= last_distance {
                break;
            }
            let at = pos - distance;
            if input[at + best_len] == input[pos + best_len] {
                let len = match_len(&input[at..], &input[pos..], max_len);
                if len > best_len {
                    best_len = len;
                    best_distance = distance;
                    if len >= nice {
                        break;
                    }
                }
            }
            chain -= 1;
            if chain == 0 {
                break;
            }
            last_distance = distance;
            candidate = self.prev[at & WINDOW_MASK];
        }
        if best_distance == 0 { (0, 0) } else { (best_len, best_distance) }
    }
}

/// Compresses `input` as one raw DEFLATE stream appended to `out`.
pub(super) fn compress(input: &[u8], level: Level, out: &mut Vec<u8>) {
    let mut w = BitWriter { out, buf: 0, cnt: 0 };
    if level == Level::Store {
        write_stored(&mut w, input, true);
        return;
    }
    let params = level.params();
    let mut chains = Chains::new();
    let mut block = Block::new();
    let n = input.len();
    let mut pos = 0;
    // The input before `emitted` is covered by the tokens written or collected so far.
    let mut emitted = 0;
    let mut block_start = 0;
    // A match found at `pos - 1` that waits to see whether `pos` has a longer one.
    let mut pending = (0_usize, 0_usize);

    while pos < n {
        let mut found = (0, 0);
        if n - pos >= MIN_MATCH {
            let candidate = chains.head[hash3(input, pos)];
            if pending.0 == 0 || pending.0 < params.max_lazy {
                let chain = if pending.0 >= params.good { params.max_chain >> 2 } else { params.max_chain };
                found = chains.longest_match(input, pos, candidate, chain, pending.0.max(MIN_MATCH - 1), params.nice);
                if found.0 == MIN_MATCH && found.1 > TOO_FAR {
                    found = (0, 0);
                }
            }
            chains.insert(input, pos);
        }
        if pending.0 != 0 {
            if found.0 > pending.0 {
                // The match one byte later is longer: the byte before it goes out as a literal.
                block.literal(input[pos - 1]);
                emitted = pos;
                pending = found;
                pos += 1;
            } else {
                let (len, distance) = pending;
                pending = (0, 0);
                block.matched(len, distance);
                let end = pos - 1 + len;
                if len <= params.max_insert {
                    chains.insert_range(input, pos + 1, end);
                }
                pos = end;
                emitted = end;
            }
        } else if found.0 != 0 {
            if found.0 < params.max_lazy {
                pending = found;
                pos += 1;
            } else {
                block.matched(found.0, found.1);
                let end = pos + found.0;
                if found.0 <= params.max_insert {
                    chains.insert_range(input, pos + 1, end);
                }
                pos = end;
                emitted = end;
            }
        } else {
            block.literal(input[pos]);
            pos += 1;
            emitted = pos;
        }
        if block.tokens.len() >= BLOCK_TOKENS {
            write_block(&mut w, &mut block, &input[block_start..emitted], false);
            block_start = emitted;
        }
    }
    // A pending match is always resolved inside the loop (it ends at or before the input's end, so
    // the loop runs again after it is found); should that ever not hold, literals are still right.
    for &byte in input.get(emitted..).unwrap_or_default() {
        block.literal(byte);
    }
    write_block(&mut w, &mut block, &input[block_start..], true);
    w.align();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kraft(lens: &[u8]) -> f64 {
        lens.iter().filter(|&&l| l != 0).map(|&l| 0.5_f64.powi(i32::from(l))).sum()
    }

    #[test]
    fn symbol_tables_invert_the_base_tables() {
        for len in MIN_MATCH..=MAX_MATCH {
            let sym = usize::from(LEN_SYM[len - MIN_MATCH]);
            let base = usize::from(LEN_BASE[sym]);
            assert!(len >= base && len - base < (1 << LEN_EXTRA[sym]), "length {len}");
        }
        assert_eq!(LEN_SYM[255], 28, "length 258 uses symbol 285");
        for distance in 1..=WINDOW {
            let sym = dist_symbol(distance);
            let base = usize::from(DIST_BASE[sym]);
            assert!(distance >= base && distance - base < (1 << DIST_EXTRA[sym]), "distance {distance}");
        }
    }

    #[test]
    fn code_lengths_are_complete_and_limited() {
        let mut rng = humanart_core::rng::Rng::new(0xC0DE);
        for case in 0..300 {
            let n = 2 + rng.index(285);
            let mut freqs = vec![0_u32; n];
            for f in &mut freqs {
                // Mostly zero or small, sometimes huge: skewed enough to need the length limit.
                *f = match rng.index(6) {
                    0 | 1 => 0,
                    2 => 1,
                    3 => rng.index(50) as u32,
                    4 => 1 << rng.index(30),
                    _ => rng.next_u32() >> 4,
                };
            }
            for limit in [7, 15] {
                if limit == 7 && n > 19 {
                    continue;
                }
                let mut lens = vec![0_u8; n];
                build_lengths(&freqs, limit, &mut lens);
                assert!(lens.iter().all(|&l| usize::from(l) <= limit), "case {case}");
                assert!((kraft(&lens) - 1.0).abs() < 1e-12, "case {case}: Kraft sum {}", kraft(&lens));
                let used = freqs.iter().filter(|&&f| f != 0).count();
                assert_eq!(lens.iter().filter(|&&l| l != 0).count(), used.max(2), "case {case}");
                for (sym, &f) in freqs.iter().enumerate() {
                    assert!(f == 0 || lens[sym] != 0, "case {case}: used symbol {sym} has no codeword");
                }
            }
        }
    }

    #[test]
    fn code_lengths_are_optimal_when_the_limit_does_not_bind() {
        // The textbook example: frequencies 5, 9, 12, 13, 16, 45 give lengths 4, 4, 3, 3, 3, 1.
        let mut lens = [0_u8; 6];
        build_lengths(&[5, 9, 12, 13, 16, 45], 15, &mut lens);
        assert_eq!(lens, [4, 4, 3, 3, 3, 1]);
        // Fibonacci frequencies make the deepest possible tree; the limit must flatten it.
        let mut fib = [0_u32; 30];
        (fib[0], fib[1]) = (1, 1);
        for i in 2..30 {
            fib[i] = fib[i - 1] + fib[i - 2];
        }
        let mut lens = [0_u8; 30];
        build_lengths(&fib, 15, &mut lens);
        assert_eq!(*lens.iter().max().unwrap(), 15);
        assert!((kraft(&lens) - 1.0).abs() < 1e-12);
        assert!(lens.windows(2).all(|w| w[0] >= w[1]), "rarer symbols never get shorter codewords: {lens:?}");
    }

    #[test]
    fn canonical_codes_are_prefix_free() {
        let lens = [3_u8, 3, 3, 3, 3, 2, 4, 4];
        let mut codes = [0_u16; 8];
        make_codes(&lens, &mut codes);
        // RFC 1951 section 3.2.2's example: F = 00, A = 010, ..., G = 1110, H = 1111 (here reversed).
        let expect = [0b010_u16, 0b110, 0b001, 0b101, 0b011, 0b00, 0b0111, 0b1111];
        assert_eq!(codes, expect);
    }

    #[test]
    fn run_length_coding_covers_every_run_shape() {
        let mut out = [(0_u8, 0_u8); NUM_LITLEN + NUM_DIST];
        let expand = |runs: &[(u8, u8)]| {
            let mut lens: Vec<u8> = Vec::new();
            for &(sym, extra) in runs {
                match sym {
                    16 => {
                        let last = *lens.last().unwrap();
                        lens.extend(std::iter::repeat_n(last, 3 + usize::from(extra)));
                    }
                    17 => lens.extend(std::iter::repeat_n(0, 3 + usize::from(extra))),
                    18 => lens.extend(std::iter::repeat_n(0, 11 + usize::from(extra))),
                    _ => lens.push(sym),
                }
            }
            lens
        };
        for run in 1..=300 {
            for value in [0_u8, 5] {
                let mut lens = vec![7_u8];
                lens.extend(std::iter::repeat_n(value, run));
                lens.push(9);
                lens.truncate(NUM_LITLEN + NUM_DIST);
                let n = run_length_code(&lens, &mut out);
                assert_eq!(expand(&out[..n]), lens, "run of {run} × {value}");
                assert!(out[..n].iter().all(|&(sym, extra)| u32::from(extra) < (1 << precode_extra(sym)).max(1)));
            }
        }
    }

    #[test]
    fn match_length_counts_equal_bytes() {
        let a: Vec<u8> = (0..300).map(|i| (i % 251) as u8).collect();
        for same in [0, 1, 7, 8, 9, 15, 16, 17, 100, 257, 258] {
            let mut b = a.clone();
            if same < b.len() {
                b[same] ^= 0x40;
            }
            assert_eq!(match_len(&a, &b, 258), same.min(258));
            assert_eq!(match_len(&a, &b, 5), same.min(5));
        }
    }
}
