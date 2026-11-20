//! The headless HumanArt command line: render, bench, export, serve and the offline tools.
//!
//! Implemented so far: `render` (`[CLI-01]`, one offscreen still through the editor's engine
//! viewer). Arguments are parsed by the in-house declarative parser of [`args`] (`[CLI-02]`);
//! errors exit with one code per class and, with `--json-errors`, print one JSON line on stderr
//! (`[CLI-03]`, [`error`]).
#![forbid(unsafe_code)]

mod args;
mod error;
mod render;
mod scan;

use std::process::ExitCode;

use args::Parsed;
use error::CliError;

const PROGRAM: &str = "humanart-cli";

fn usage() -> String {
    format!(
        "{PROGRAM}: the headless HumanArt command line.\n\nUsage: {PROGRAM} <command> [options]\n\nCommands:\n  render  {}\n  render-mesh  {}\n  pick-landmarks  {}\n  fit-scan  {}\n  face-model  {}\n\n{PROGRAM} <command> --help lists a command's options.\n",
        render::RENDER.summary,
        scan::RENDER_MESH.summary,
        scan::PICK_LANDMARKS.summary,
        scan::FIT_SCAN.summary,
        scan::FACE_MODEL.summary
    )
}

fn run(args: &[String]) -> Result<(), CliError> {
    match args.first().map(String::as_str) {
        None | Some("--help" | "-h" | "help") => {
            print!("{}", usage());
            Ok(())
        }
        Some("render") => {
            let parsed = Parsed::parse(&render::RENDER, args.get(1..).unwrap_or(&[]))?;
            if parsed.help() {
                print!("{}", args::help(PROGRAM, &render::RENDER));
                return Ok(());
            }
            let job = render::RenderJob::from_args(&parsed)?;
            let summary = render::run(&job)?;
            println!(
                "wrote {} (template: {}, adapter: {}, skin surface: {})",
                summary.out.display(),
                summary.template,
                summary.adapter,
                if summary.skin_surface { "on" } else { "off" }
            );
            if !summary.missing_names.is_empty() {
                println!("skin zones without their template names: {}", summary.missing_names.join(", "));
            }
            Ok(())
        }
        Some("render-mesh") => scan::command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("pick-landmarks") => scan::pick_command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("fit-scan") => scan::fit_command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("face-model") => scan::face_model_command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some(other) => Err(CliError::Usage(args::UsageError(format!("unknown command '{other}'; try --help")))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json-errors");
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if json {
                eprintln!("{}", e.to_json());
            } else {
                eprintln!("{PROGRAM}: {e}");
            }
            ExitCode::from(e.exit_code())
        }
    }
}
