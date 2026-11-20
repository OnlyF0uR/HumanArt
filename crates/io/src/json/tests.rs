//! Cross-module JSON tests: RFC 8259 conformance cases written out by hand (after the accept / reject
//! sets of the well-known JSON parsing test suites), round trips through every writer style, the
//! `ToJson` / `FromJson` pair, and the mutation fuzz `[AC-WP03-1]` `[AC-WP03-2]` `[ROB-01]`.

use super::*;
use humanart_core::fuzz;
use humanart_core::rng::Rng;
use std::time::Duration;

/// Texts every conforming parser accepts.
const ACCEPT: &[&[u8]] = &[
    // Structure and white space.
    b"[]",
    b"{}",
    b"[[]   ]",
    b" [] ",
    b"[1\n]",
    b" [1]",
    b"[2] ",
    b"[ 4]",
    b"\t\r\n[\t\r\n1\t\r\n,\t\r\n2\t\r\n]\t\r\n",
    b"[null, 1, \"1\", {}]",
    b"[1,null,null,null,2]",
    b"{\"asd\":\"sdf\", \"dfg\":\"fgh\"}",
    b"{\"\":0}",
    b"{\"x\":[{\"id\": \"xxx\"}], \"id\": \"xxx\"}",
    b"{\"a\":[]}",
    b"{\n\"a\": \"b\"\n}",
    b"[[[[[[[[[[[[[[[[[[[[\"deep\"]]]]]]]]]]]]]]]]]]]]",
    // Scalars at the top level.
    b"false",
    b"true",
    b"null",
    b"42",
    b"-0.1",
    b"\"asd\"",
    b"\"\"",
    b"\" \"",
    // Numbers.
    b"[0]",
    b"[-0]",
    b"[-1]",
    b"[-123]",
    b"[123]",
    b"[123.456789]",
    b"[0e1]",
    b"[0e+1]",
    b"[0E-1]",
    b"[20e1]",
    b"[1E22]",
    b"[1E-2]",
    b"[1E+2]",
    b"[1e-2]",
    b"[1e+2]",
    b"[123e65]",
    b"[123e45]",
    b"[123.456e78]",
    b"[-0.000000000000000000000000000000000000000000000000000000000000000000000000000001]\n",
    b"[1.7976931348623157e308]",
    b"[4.9e-324]",
    b"[1e-400]",
    b"[9223372036854775807, -9223372036854775808, 18446744073709551615, 18446744073709551616]",
    b"{ \"min\": -1.0e+28, \"max\": 1.0e+28 }",
    // Strings and escapes.
    b"[\"\"]",
    b"[\"a/*b*/c/*d//e\"]",
    b"[\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"]",
    b"[\"\\u0060\\u012a\\u12AB\"]",
    b"[\"\\uD801\\udc37\"]",
    b"[\"\\ud83d\\ude39\\ud83d\\udc8d\"]",
    b"[\"\\uDBFF\\uDFFF\"]",
    b"[\"\\u0000\"]",
    b"{\"foo\\u0000bar\": 42}",
    b"[\"\\\\u0000\"]",
    b"[\"\\u0012\"]",
    b"[\"\\uFFFF\"]",
    b"[\"\\uFFFE\"]",
    b"[\"\\uFDD0\"]",
    b"[\"\\u2028\\u2029\"]",
    b"[\"\\u005C\"]",
    b"[\"\\u0022\"]",
    b"[\"new\\u000Aline\"]",
    b"[\"new\\u00A0line\"]",
    b"[\"\\u0061\\u30af\\u30EA\\u30b9\"]",
    b"{\"title\":\"\\u041f\\u043e\\u043b\\u0442\\u043e\\u0440\\u0430\"}",
    "[\"\u{7F}\u{80}\u{7FF}\u{800}\u{FFFF}\u{10000}\u{10FFFF}\"]".as_bytes(),
    "[\"\u{2028}\u{2029}\"]".as_bytes(),
    "[\"π €𝄞 ⍂㈴⍂\"]".as_bytes(),
    // A byte order mark before the text is skipped.
    b"\xEF\xBB\xBF{}",
];

/// Texts a strict RFC 8259 parser refuses. The comment names the class of each group.
const REJECT: &[&[u8]] = &[
    // Empty and stray input.
    b"",
    b" ",
    b"\xEF\xBB\xBF",
    b"\xEF\xBB{}",
    b"]",
    b"}",
    b"1]",
    b"[1]]",
    b"[1]x",
    b"{}}",
    b"[][]",
    b"{\"a\": true} \"x\"",
    b"2@",
    b"<.>",
    b"[<null>]",
    b"*",
    b"\x00",
    b"[\x00]",
    // Unclosed containers.
    b"[",
    b"[1",
    b"[1,",
    b"[{}",
    b"{",
    b"{\"a\"",
    b"{\"a\":",
    b"{\"a\":1",
    b"{\"asd\":\"asd\"",
    b"[\"a",
    b"[\"a\"",
    b"[\"\\{[\"\\{[\"\\{",
    // Separators.
    b"[1 true]",
    b"[,1]",
    b"[1,,2]",
    b"[\"x\",,]",
    b"[,]",
    b"[   , \"\"]",
    b"[1:2]",
    b"[\"\": 1]",
    b"[\"\"],",
    b"[3[4]]",
    // Trailing commas.
    b"[1,]",
    b"[\"\",]",
    b"[1,,]",
    b"{\"a\":\"b\",}",
    b"{\"id\":0,}",
    b"{\"id\":0,,,,,}",
    b"{\"a\":\"b\",,\"c\":\"d\"}",
    // Objects.
    b"{\"x\", null}",
    b"{\"x\"::\"b\"}",
    b"{\"a\" b}",
    b"{\"a\" \"b\"}",
    b"{:\"b\"}",
    b"{1:1}",
    b"{null:null}",
    b"{key: \"value\"}",
    b"{'a':0}",
    b"{\"a\":\"a\" 123}",
    b"{ \"foo\" : \"bar\", \"a\" }",
    b"{]",
    b"{,",
    b"{[",
    b"{\"a\":\"b\"}}",
    // Duplicate keys: legal by the RFC's grammar, refused here (see the parser module comment).
    b"{\"a\":\"b\",\"a\":\"c\"}",
    b"{\"a\":1,\"b\":2,\"a\":1}",
    // Comments.
    b"{\"a\":\"b\"}/**/",
    b"{\"a\":\"b\"}//",
    b"{\"a\":/*comment*/\"b\"}",
    b"{\"a\":\"b\"}#",
    b"[1] // trailing",
    // Literals.
    b"nul",
    b"tru",
    b"fals",
    b"[nulL]",
    b"[True]",
    b"[ false, nul",
    b"[ true, fals",
    b"[ false, tru",
    b"[NaN]",
    b"[-NaN]",
    b"[Inf]",
    b"[+Inf]",
    b"[Infinity]",
    b"[-Infinity]",
    b"abc",
    // Numbers.
    b"[++1234]",
    b"[+1]",
    b"[-]",
    b"[- 1]",
    b"[-foo]",
    b"[-01]",
    b"[-012]",
    b"[012]",
    b"[00]",
    b"[-1.0.]",
    b"[0.1.2]",
    b"[-2.]",
    b"[1.]",
    b"[.2e-3]",
    b"[.123]",
    b"[-.123]",
    b"[.-1]",
    b"[0.e1]",
    b"[2.e3]",
    b"[2.e+3]",
    b"[2.e-3]",
    b"[9.e+]",
    b"[0.3e]",
    b"[0.3e+]",
    b"[0E]",
    b"[0E+]",
    b"[0e+-1]",
    b"[1.0e]",
    b"[1.0e+]",
    b"[1.0e-]",
    b"[1eE2]",
    b"[1ea]",
    b"[1.2a-3]",
    b"[1 000.0]",
    b"[1+2]",
    b"[0x1]",
    b"[0x42]",
    b"[-1x]",
    b"[-123.123foo]",
    b"[1.8011670033376514H-308]",
    b"[123\xe5]",
    b"[1e1\xe5]",
    b"[0\xe5]",
    "[\u{FF11}]".as_bytes(),
    // Too large for an f64 (see the number module comment).
    b"[1e400]",
    b"[-1e309]",
    b"{9999E9999:1}",
    // Strings.
    b"\"",
    b"\"no closing",
    b"['single quote']",
    b"[\"\\\"]",
    b"[\"\\",
    b"[\"\\a\"]",
    b"[\"\\x00\"]",
    b"[\"\\UA66D\"]",
    b"[\"\\u00A\"]",
    b"[\"\\uqqqq\"]",
    b"[\"\\uD834\\uDd\"]",
    b"[\\u0020\"asd\"]",
    b"[\\n]",
    "[\"\\\u{1F300}\"]".as_bytes(),
    // Raw control characters inside a string.
    b"[\"a\x00a\"]",
    b"[\"a\ta\"]",
    b"[\"new\nline\"]",
    b"[\"\x1f\"]",
    // Lone and broken surrogates.
    b"[\"\\uD800\"]",
    b"[\"\\uDd1ea\"]",
    b"[\"\\uD800\\\"]",
    b"[\"\\uD800\\u\"]",
    b"[\"\\uD800\\u1\"]",
    b"[\"\\uD800\\u1x\"]",
    b"[\"\\uD800\\uD800\"]",
    b"[\"\\uD800\\n\"]",
    // Invalid UTF-8, in and out of strings.
    b"[\"\xff\"]",
    b"[\"\xc0\xaf\"]",
    b"[\"\xed\xa0\x80\"]",
    b"[\"\xe0\xff\"]",
    b"[\"\xf4\x90\x80\x80\"]",
    b"[\"a\xe9\"]",
    b"[\xff]",
    b"[a\xe5]",
    b"\xe9",
    // White space RFC 8259 does not allow.
    b"[\x0c]",
    b"[\x0b1]",
    "[\u{A0}1]".as_bytes(),
    "[\u{2060}]".as_bytes(),
    "[1\u{3000}]".as_bytes(),
];

#[test]
fn ac_wp03_1_conformance_accept_set() {
    for (i, text) in ACCEPT.iter().enumerate() {
        let value = parse(text).unwrap_or_else(|e| panic!("accept case {i} {:?}: {e}", String::from_utf8_lossy(text)));
        // Every accepted text survives a write and a re-read unchanged.
        for style in [Style::COMPACT, Style::PRETTY, Style::PRETTY_INLINE] {
            let mut out = String::new();
            write(&value, style, &mut out).unwrap();
            assert_eq!(parse_str(&out).unwrap(), value, "accept case {i} via {out}");
        }
    }
}

#[test]
fn ac_wp03_1_conformance_reject_set() {
    for (i, text) in REJECT.iter().enumerate() {
        if let Ok(value) = parse(text) {
            panic!("reject case {i} {:?} was accepted as {value:?}", String::from_utf8_lossy(text));
        }
    }
}

#[test]
fn accepted_values_are_the_expected_ones() {
    assert_eq!(parse(b"[-0]").unwrap()[0].as_number().map(|n| n.as_f64().to_bits()), Some((-0.0_f64).to_bits()));
    assert_eq!(parse(b"[1e-400]").unwrap()[0], Value::from(0.0));
    let ints =
        parse(b"[9223372036854775807, -9223372036854775808, 18446744073709551615, 18446744073709551616]").unwrap();
    assert_eq!(ints[0], Value::from(i64::MAX));
    assert_eq!(ints[1], Value::from(i64::MIN));
    assert_eq!(ints[2], Value::from(u64::MAX));
    assert_eq!(ints[3], Value::from(18_446_744_073_709_551_616.0));
    assert_eq!(parse(b"[\"\\uD801\\udc37\"]").unwrap()[0], Value::from("\u{10437}"));
    assert_eq!(parse(b"{\"foo\\u0000bar\": 42}").unwrap()["foo\0bar"], Value::from(42));
    assert_eq!(parse(b"\xEF\xBB\xBF\"x\"").unwrap(), Value::from("x"));
}

/// A random value tree: every kind, strings with escapes and non-ASCII text, floats from random bit
/// patterns, integers across both ranges.
fn random_value(rng: &mut Rng, depth: u32) -> Value {
    let pick = if depth == 0 { rng.index(5) } else { rng.index(7) };
    match pick {
        0 => Value::Null,
        1 => Value::Bool(rng.chance(0.5)),
        2 => match rng.index(4) {
            0 => Value::from(rng.next_u64() as i64),
            1 => Value::from(rng.next_u64()),
            2 => Value::from(rng.range_i64(-1000, 1000)),
            _ => {
                let f = f64::from_bits(rng.next_u64());
                Value::from(if f.is_finite() { f } else { rng.normal() })
            }
        },
        3 | 4 => Value::from(random_string(rng)),
        5 => (0..rng.index(6)).map(|_| random_value(rng, depth - 1)).collect(),
        _ => {
            let mut object = Object::new();
            for _ in 0..rng.index(6) {
                object.insert(random_string(rng), random_value(rng, depth - 1));
            }
            Value::from(object)
        }
    }
}

fn random_string(rng: &mut Rng) -> String {
    const PIECES: &[&str] = &[
        "a", "Z", "0", " ", "\"", "\\", "/", "\n", "\t", "\u{0}", "\u{1f}", "\u{7f}", "é", "€", "𝄞", "\u{2028}", "key",
    ];
    (0..rng.index(8)).map(|_| PIECES[rng.index(PIECES.len())]).collect()
}

#[test]
fn ac_wp03_1_random_trees_round_trip_through_every_style() {
    let mut rng = Rng::new(0x15_0A_7A);
    let styles = [
        Style::COMPACT,
        Style::PRETTY,
        Style::PRETTY_INLINE,
        Style { indent: Indent::Tab, inline_scalar_arrays: false },
        Style { indent: Indent::Spaces(7), inline_scalar_arrays: true },
    ];
    for case in 0..2000 {
        let value = random_value(&mut rng, 5);
        let compact = to_string(&value).unwrap();
        for style in styles {
            let mut text = String::new();
            write(&value, style, &mut text).unwrap();
            let back = parse_str(&text).unwrap_or_else(|e| panic!("case {case}: {e} in {text}"));
            assert_eq!(back, value, "case {case}");
            // Writing is deterministic: the re-read value writes the same bytes.
            assert_eq!(to_string(&back).unwrap(), compact, "case {case}");
        }
    }
}

#[test]
fn canonical_output_does_not_depend_on_member_order() {
    let a = parse_str(r#"{"b": {"y": 1, "x": [ {"q": 1, "p": 2} ]}, "a": null}"#).unwrap();
    let b = parse_str(r#"{"a": null, "b": {"x": [{"p": 2, "q": 1}], "y": 1}}"#).unwrap();
    assert_ne!(to_string(&a).unwrap(), to_string(&b).unwrap());
    let (mut a, mut b) = (a, b);
    a.sort_keys_deep();
    b.sort_keys_deep();
    assert_eq!(to_string(&a).unwrap(), r#"{"a":null,"b":{"x":[{"p":2,"q":1}],"y":1}}"#);
    assert_eq!(to_string(&a).unwrap(), to_string(&b).unwrap());
}

/// A type written with `ToJson` and read with `FromJson`, as the project's own formats are.
#[derive(Debug, PartialEq)]
struct Rig {
    name: String,
    joints: Vec<Joint>,
    scale: f32,
    flags: Option<u8>,
}

#[derive(Debug, PartialEq)]
struct Joint {
    parent: Option<u16>,
    rest: [f32; 4],
}

impl ToJson for Joint {
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> crate::Result<()> {
        w.begin_object()?.field("parent", &self.parent)?.field("rest", &self.rest)?.end_object()?;
        Ok(())
    }
}

impl FromJson<'_> for Joint {
    fn from_json(node: &Node<'_, '_>) -> Result<Self, AccessError> {
        node.deny_unknown(&["parent", "rest"])?;
        Ok(Self { parent: node.get("parent")?, rest: node.get("rest")? })
    }
}

impl ToJson for Rig {
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> crate::Result<()> {
        w.begin_object()?.field("name", &self.name)?.field("joints", &self.joints)?.field("scale", &self.scale)?;
        w.opt_field("flags", self.flags.as_ref())?.end_object()?;
        Ok(())
    }
}

impl FromJson<'_> for Rig {
    fn from_json(node: &Node<'_, '_>) -> Result<Self, AccessError> {
        node.deny_unknown(&["name", "joints", "scale", "flags"])?;
        Ok(Self {
            name: node.get("name")?,
            joints: node.get("joints")?,
            scale: node.get("scale")?,
            flags: node.get_opt("flags")?,
        })
    }
}

#[test]
fn typed_values_round_trip_through_to_json_and_from_json() {
    let rig = Rig {
        name: "body".into(),
        joints: vec![
            Joint { parent: None, rest: [0.0, 0.0, 0.0, 1.0] },
            Joint { parent: Some(0), rest: [0.1, -0.2, 0.3, 0.927_362] },
        ],
        scale: 0.01,
        flags: None,
    };
    let compact = stringify(&rig, Style::COMPACT).unwrap();
    assert_eq!(
        compact,
        r#"{"name":"body","joints":[{"parent":null,"rest":[0.0,0.0,0.0,1.0]},{"parent":0,"rest":[0.1,-0.2,0.3,0.927362]}],"scale":0.01}"#
    );
    let pretty = stringify(&rig, Style::PRETTY_INLINE).unwrap();
    assert!(pretty.contains("\"rest\": [0.1, -0.2, 0.3, 0.927362]"), "{pretty}");
    for text in [&compact, &pretty] {
        let value = parse_str(text).unwrap();
        assert_eq!(value.node().read::<Rig>().unwrap(), rig);
    }
    let with_flags = Rig { flags: Some(3), ..rig };
    let value = parse_str(&stringify(&with_flags, Style::COMPACT).unwrap()).unwrap();
    assert_eq!(value.node().read::<Rig>().unwrap(), with_flags);
    // Scalars, strings, slices and values all write through the same trait.
    assert_eq!(stringify("a\"b", Style::COMPACT).unwrap(), r#""a\"b""#);
    assert_eq!(stringify(&[1_u64 << 63, 7], Style::COMPACT).unwrap(), "[9223372036854775808,7]");
    assert_eq!(stringify(&[[1_i8, -2], [3, 4]], Style::PRETTY_INLINE).unwrap(), "[\n  [1, -2],\n  [3, 4]\n]");
    assert_eq!(stringify(&value, Style::COMPACT).unwrap(), to_string(&value).unwrap());
    assert!(stringify(&[f32::NAN], Style::COMPACT).is_err());
    assert!(stringify(&Vec::<Option<bool>>::new(), Style::PRETTY).is_ok_and(|t| t == "[]"));
}

#[test]
fn rob_01_mutation_fuzz() {
    let mut corpus: Vec<Vec<u8>> = ACCEPT.iter().map(|t| t.to_vec()).collect();
    let mut rng = Rng::new(0xF0_22);
    for _ in 0..12 {
        corpus.push(to_string_pretty(&random_value(&mut rng, 4)).unwrap().into_bytes());
    }
    corpus.push(b"{\"lods\":[{\"name\":\"a\",\"vertex_count\":12,\"scale\":[1.0,2.5e-3,-7]}],\"n\":null}".to_vec());
    fuzz::run("json", &corpus, 400, 0x15_0A_7A, Duration::from_secs(120), |bytes| {
        if let Ok(value) = parse_with(bytes, &JsonLimits::FUZZ) {
            // Whatever is accepted writes and re-reads to the same value, and typed access on it
            // never panics.
            let text = to_string(&value).unwrap_or_default();
            assert_eq!(parse_str(&text).ok(), Some(value.clone()));
            let node = value.node();
            let _ = node.get::<Vec<f32>>("scale");
            let first_scale = || node.field("lods")?.at(0)?.get::<[f64; 3]>("scale");
            let _ = first_scale();
            let _ = node.read::<Vec<Option<String>>>();
            let _ = node.members().map(|m| m.map(|(_, n)| n.path().len()).sum::<usize>());
        }
    });
}
