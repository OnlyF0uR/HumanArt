//! The headless HumanArt command line: render, bench, export, serve and the offline tools.
//!
//! Implemented so far: `render` (`[CLI-01]`, one offscreen still through the editor's engine
//! viewer), `turntable` (stills around the character), `serve` (the agent server on stdio or
//! loopback HTTP, `[AG-01]`), `batch` (a job file of tool calls), `validate` (data and project
//! checks), `export` (the character as glTF, `[EXP-P1]`), `bench` (scenario 10, the UI benchmark scene `[UI-10]`), `fontbake` (TrueType fonts to
//! the UI's `.anfont` atlases `[UI-03]`) and the scan tools. Arguments are parsed by the in-house
//! declarative parser of [`args`] (`[CLI-02]`); errors exit with one code per class and, with
//! `--json-errors`, print one JSON line on stderr (`[CLI-03]`, [`error`]).
#![forbid(unsafe_code)]

mod args;
mod batch;
mod bench;
mod error;
mod export;
mod face_detail;
mod fontbake;
mod host;
mod render;
mod scan;
mod serve;
mod skin_detail;
mod turntable;
mod validate;

use std::process::ExitCode;

use args::{CommandSpec, Parsed};
use error::CliError;

const PROGRAM: &str = "humanart-cli";

// [PERF-R1] The unit-test binary counts allocations, so the test of `bench` shows the UI scene's
// allocation count through the command's path. The shipped binary keeps the system allocator.
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

fn usage() -> String {
    format!(
        "{PROGRAM}: the headless HumanArt command line.\n\nUsage: {PROGRAM} <command> [options]\n\nCommands:\n  render  {}\n  turntable  {}\n  serve  {}\n  batch  {}\n  validate  {}\n  export  {}\n  bench  {}\n  fontbake  {}\n  render-mesh  {}\n  pick-landmarks  {}\n  fit-scan  {}\n  face-model  {}\n  skin-detail  {}\n  face-detail  {}\n\n{PROGRAM} <command> --help lists a command's options.\nExit codes: 0 success, 2 usage, 3 data, 4 GPU, 5 render, 6 file, 7 project, 8 job file,\n9 failed steps or checks, 10 agent server.\n",
        render::RENDER.summary,
        turntable::TURNTABLE.summary,
        serve::SERVE.summary,
        batch::BATCH.summary,
        validate::VALIDATE.summary,
        export::EXPORT.summary,
        bench::BENCH.summary,
        fontbake::FONTBAKE.summary,
        scan::RENDER_MESH.summary,
        scan::PICK_LANDMARKS.summary,
        scan::FIT_SCAN.summary,
        scan::FACE_MODEL.summary,
        skin_detail::SKIN_DETAIL.summary,
        face_detail::FACE_DETAIL.summary
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
        Some("turntable") => {
            let Some(parsed) = command(&turntable::TURNTABLE, args)? else { return Ok(()) };
            let summary = turntable::run(&turntable::TurntableJob::from_args(&parsed)?)?;
            println!("wrote {} stills (adapter: {})", summary.files.len(), summary.adapter);
            Ok(())
        }
        Some("serve") => {
            let Some(parsed) = command(&serve::SERVE, args)? else { return Ok(()) };
            serve::run(&serve::ServeJob::from_args(&parsed)?)
        }
        Some("batch") => {
            let Some(parsed) = command(&batch::BATCH, args)? else { return Ok(()) };
            let report = batch::run(&batch::BatchJob::from_args(&parsed)?)?;
            println!("{}", batch::report_text(&report)?);
            if report.failed > 0 {
                return Err(CliError::Failed(format!("{} of {} steps failed", report.failed, report.ran)));
            }
            Ok(())
        }
        Some("validate") => {
            let Some(parsed) = command(&validate::VALIDATE, args)? else { return Ok(()) };
            let job = validate::ValidateJob::from_args(&parsed);
            let findings = validate::run(&job);
            print!("{}", validate::report(&findings, job.json)?);
            validate::verdict(&findings)
        }
        Some("export") => {
            let Some(parsed) = command(&export::EXPORT, args)? else { return Ok(()) };
            let job = export::ExportJob::from_args(&parsed)?;
            let r = export::run(&job)?;
            println!(
                "wrote {}: {} primitives, {} vertices, {} triangles, {} joints, {} morph targets",
                job.out.display(),
                r.primitives,
                r.vertices,
                r.triangles,
                r.joints,
                r.targets
            );
            Ok(())
        }
        Some("bench") => {
            let Some(parsed) = command(&bench::BENCH, args)? else { return Ok(()) };
            let job = bench::BenchJob::from_args(&parsed)?;
            let summary = bench::run(&job)?;
            match &job.out {
                Some(out) => println!("{}\nwrote {}", summary.line(), out.display()),
                None => println!("{}", summary.to_json()?),
            }
            Ok(())
        }
        Some("fontbake") => {
            let Some(parsed) = command(&fontbake::FONTBAKE, args)? else { return Ok(()) };
            fontbake::run(&fontbake::FontbakeJob::from_args(&parsed)?, |baked| println!("{}", baked.line()))?;
            Ok(())
        }
        Some("render-mesh") => scan::command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("pick-landmarks") => scan::pick_command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("fit-scan") => scan::fit_command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("face-model") => scan::face_model_command(PROGRAM, args.get(1..).unwrap_or(&[])),
        Some("face-detail") => {
            let Some(parsed) = command(&face_detail::FACE_DETAIL, args)? else { return Ok(()) };
            let job = face_detail::FaceDetailJob::from_args(&parsed)?;
            let (share, sd) = face_detail::run(&job)?;
            println!(
                "wrote {} ({}²; {:.1} % of the tile used; log-ratio SD r {:.3}, g {:.3}, b {:.3})",
                job.out.display(),
                job.size,
                100.0 * share,
                sd[0],
                sd[1],
                sd[2]
            );
            Ok(())
        }
        Some("skin-detail") => {
            let Some(parsed) = command(&skin_detail::SKIN_DETAIL, args)? else { return Ok(()) };
            let job = skin_detail::SkinDetailJob::from_args(&parsed)?;
            let sd = skin_detail::run(&job)?;
            println!(
                "wrote {} ({}² texels; log-ratio SD r {:.3}, g {:.3}, b {:.3})",
                job.out.display(),
                job.rect[2],
                sd[0],
                sd[1],
                sd[2]
            );
            Ok(())
        }
        Some(other) => Err(CliError::Usage(args::UsageError(format!("unknown command '{other}'; try --help")))),
    }
}

/// Parses `args` (the command name first) against `spec`; `None` after printing the help.
fn command(spec: &'static CommandSpec, args: &[String]) -> Result<Option<Parsed>, CliError> {
    let parsed = Parsed::parse(spec, args.get(1..).unwrap_or(&[]))?;
    if parsed.help() {
        print!("{}", args::help(PROGRAM, spec));
        return Ok(None);
    }
    Ok(Some(parsed))
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
