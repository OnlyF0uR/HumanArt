//! The in-house declarative argument parser `[CLI-02]`: a command lists its options once
//! ([`OptSpec`]); parsing, defaults, unknown-option errors and the `--help` text all come from that
//! list.

use std::fmt::Write as _;

/// One `--name` option of a command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptSpec {
    /// The option's name without the leading `--`.
    pub name: &'static str,
    /// The value's placeholder in the help (`None`: a flag without a value).
    pub value: Option<&'static str>,
    /// The default value shown in the help and used when the option is absent.
    pub default: Option<&'static str>,
    /// One line of help.
    pub help: &'static str,
}

/// A command: its name, a summary and its options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    /// The command name.
    pub name: &'static str,
    /// One line of help.
    pub summary: &'static str,
    /// The options.
    pub options: &'static [OptSpec],
}

/// The parsed options of one command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    spec: &'static CommandSpec,
    values: Vec<Option<String>>,
    help: bool,
}

/// A command line that does not fit its command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageError(pub String);

impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UsageError {}

impl Parsed {
    /// Parses `args` (the arguments after the command name) against `spec`: `--name value`,
    /// `--name=value` and `--flag`; `--help` or `-h` asks for the help.
    ///
    /// # Errors
    /// [`UsageError`] for an unknown option, a positional argument, a missing value or an option
    /// given twice.
    pub fn parse(spec: &'static CommandSpec, args: &[String]) -> Result<Self, UsageError> {
        let mut values = vec![None; spec.options.len()];
        let mut help = false;
        let mut i = 0;
        while let Some(arg) = args.get(i) {
            i += 1;
            if arg == "--help" || arg == "-h" {
                help = true;
                continue;
            }
            let Some(body) = arg.strip_prefix("--") else {
                return Err(UsageError(format!("{}: unexpected argument '{arg}'", spec.name)));
            };
            let (name, inline) = match body.split_once('=') {
                Some((n, v)) => (n, Some(v.to_owned())),
                None => (body, None),
            };
            let Some(index) = spec.options.iter().position(|o| o.name == name) else {
                return Err(UsageError(format!("{}: unknown option '--{name}'", spec.name)));
            };
            let opt = spec.options[index];
            let value = match (opt.value, inline) {
                (None, None) => "true".to_owned(),
                (None, Some(_)) => return Err(UsageError(format!("--{name} takes no value"))),
                (Some(_), Some(v)) => v,
                (Some(_), None) => {
                    let v = args.get(i).ok_or_else(|| UsageError(format!("--{name} needs a value")))?;
                    i += 1;
                    v.clone()
                }
            };
            if values[index].replace(value).is_some() {
                return Err(UsageError(format!("--{name} is given twice")));
            }
        }
        Ok(Self { spec, values, help })
    }

    /// Whether `--help` was asked for.
    pub const fn help(&self) -> bool {
        self.help
    }

    /// The value of option `name`, or its default (`None` when neither exists or the option is
    /// not in the command's list).
    pub fn get(&self, name: &str) -> Option<&str> {
        let index = self.spec.options.iter().position(|o| o.name == name)?;
        self.values.get(index).and_then(Option::as_deref).or(self.spec.options[index].default)
    }

    /// The value of `name` as a whole number in `min..=max`.
    ///
    /// # Errors
    /// [`UsageError`] when it is missing, not a number or out of range.
    pub fn number(&self, name: &str, min: u32, max: u32) -> Result<u32, UsageError> {
        let text = self.get(name).ok_or_else(|| UsageError(format!("--{name} is required")))?;
        let n: u32 = text.parse().map_err(|_| UsageError(format!("--{name}: '{text}' is not a whole number")))?;
        if (min..=max).contains(&n) {
            Ok(n)
        } else {
            Err(UsageError(format!("--{name}: {n} is outside {min}..={max}")))
        }
    }

    /// The value of `name`, which must be one of `choices` (returned as its index).
    ///
    /// # Errors
    /// [`UsageError`] when it is missing or not one of the choices.
    pub fn choice(&self, name: &str, choices: &[&str]) -> Result<usize, UsageError> {
        let text = self.get(name).ok_or_else(|| UsageError(format!("--{name} is required")))?;
        choices
            .iter()
            .position(|c| *c == text)
            .ok_or_else(|| UsageError(format!("--{name}: '{text}' is not one of {}", choices.join(", "))))
    }
}

/// The `--help` text of `spec`, generated from its option list.
pub fn help(program: &str, spec: &CommandSpec) -> String {
    let mut out = format!("{program} {}: {}\n\nOptions:\n", spec.name, spec.summary);
    let width = spec.options.iter().map(|o| o.name.len() + o.value.map_or(0, |v| v.len() + 1)).max().unwrap_or(0);
    for o in spec.options {
        let left = match o.value {
            Some(v) => format!("{} {v}", o.name),
            None => o.name.to_owned(),
        };
        let default = o.default.map(|d| format!(" (default: {d})")).unwrap_or_default();
        let _ = writeln!(out, "  --{left:width$}  {}{default}", o.help);
    }
    let _ = writeln!(out, "  --{:width$}  This help.", "help");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTIONS: [OptSpec; 3] = [
        OptSpec { name: "view", value: Some("NAME"), default: Some("body"), help: "What to frame." },
        OptSpec { name: "width", value: Some("PX"), default: Some("640"), help: "Width." },
        OptSpec { name: "json-errors", value: None, default: None, help: "JSON errors." },
    ];
    static SPEC: CommandSpec = CommandSpec { name: "render", summary: "Renders a still.", options: &OPTIONS };

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn cli_02_options_defaults_and_flags() {
        let p = Parsed::parse(&SPEC, &args(&["--view", "head", "--width=320", "--json-errors"])).unwrap();
        assert_eq!(p.get("view"), Some("head"));
        assert_eq!(p.number("width", 1, 8192).unwrap(), 320);
        assert_eq!(p.get("json-errors"), Some("true"));
        assert!(!p.help());
        let d = Parsed::parse(&SPEC, &[]).unwrap();
        assert_eq!(d.get("view"), Some("body"));
        assert_eq!(d.number("width", 1, 8192).unwrap(), 640);
        assert_eq!(d.get("json-errors"), None);
        assert_eq!(d.choice("view", &["head", "body"]).unwrap(), 1);
        assert!(Parsed::parse(&SPEC, &args(&["-h"])).unwrap().help());
    }

    #[test]
    fn cli_02_bad_command_lines_are_usage_errors() {
        for bad in [&["--nope"][..], &["stray"], &["--view"], &["--json-errors=1"], &["--view", "a", "--view", "b"]] {
            assert!(Parsed::parse(&SPEC, &args(bad)).is_err(), "{bad:?}");
        }
        let p = Parsed::parse(&SPEC, &args(&["--width", "0", "--view", "side"])).unwrap();
        assert!(p.number("width", 1, 8192).is_err());
        assert!(p.choice("view", &["head", "body"]).is_err());
        let p = Parsed::parse(&SPEC, &args(&["--width", "abc"])).unwrap();
        assert!(p.number("width", 1, 8192).is_err());
    }

    #[test]
    fn cli_02_help_lists_every_option_with_its_default() {
        let text = help("humanart-cli", &SPEC);
        for o in &OPTIONS {
            assert!(text.contains(&format!("--{}", o.name)), "{text}");
        }
        assert!(text.contains("(default: body)") && text.contains("--help"));
    }
}
