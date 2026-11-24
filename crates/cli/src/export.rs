//! `humanart-cli export` (WP-29, `[EXP-P1]`): the character as a glTF 2.0 binary (`.glb`), through
//! `humanart_model::export`: the rest shape with the document's identity values (registry defaults
//! and `--set`), the skeleton, the skin and the expressions as morph targets. No GPU is needed; the
//! skin textures are not part of it (they are baked on the GPU).

use std::path::PathBuf;
use std::sync::Arc;

use humanart_agent::engine::Model;
use humanart_doc::registry::ParamValue;
use humanart_model::export::{ExportOptions, ExportReport, RestShape, character_gltf};

use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;
use crate::render::{default_document, load_registry, load_template, set_values};

const OPTIONS: [OptSpec; 6] = [
    OptSpec { name: "out", value: Some("FILE"), default: Some("character.glb"), help: "The .glb file to write." },
    OptSpec {
        name: "set",
        value: Some("ID=V,..."),
        default: None,
        help: "Scalar parameter values before the export (e.g. identity.sex=1,body.height=0.5).",
    },
    OptSpec {
        name: "expressions",
        value: Some("ON"),
        default: Some("on"),
        help: "on: every expression target as a morph target; off: the shape alone.",
    },
    OptSpec { name: "name", value: Some("NAME"), default: Some("character"), help: "The scene's and mesh's name." },
    OptSpec { name: "data", value: Some("DIR"), default: Some("data"), help: "The data folder." },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `export` command.
pub static EXPORT: CommandSpec =
    CommandSpec { name: "export", summary: "Writes the character as a glTF 2.0 binary (.glb).", options: &OPTIONS };

/// What `export` was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportJob {
    /// The file to write.
    pub out: PathBuf,
    /// Scalar parameter values (registry id and the value's text).
    pub params: Vec<(String, String)>,
    /// Whether the expressions become morph targets.
    pub expressions: bool,
    /// The scene's and mesh's name.
    pub name: String,
    /// The data folder.
    pub data: PathBuf,
}

impl ExportJob {
    /// The job of a parsed `export` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] for values that are not among the choices.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let name = p.get("name").unwrap_or("character").trim();
        if name.is_empty() || name.len() > 128 {
            return Err(CliError::Usage(crate::args::UsageError("--name: 1 to 128 characters".into())));
        }
        Ok(Self {
            out: PathBuf::from(p.get("out").unwrap_or("character.glb")),
            params: set_values(p.get("set").unwrap_or(""))?,
            expressions: p.choice("expressions", &["on", "off"])? == 0,
            name: name.to_owned(),
            data: PathBuf::from(p.get("data").unwrap_or("data")),
        })
    }
}

/// Runs `job` and returns what was written.
///
/// # Errors
/// [`CliError::Data`] for the registry, the template, an unknown parameter or a refused export,
/// [`CliError::Io`] for the output file.
pub fn run(job: &ExportJob) -> Result<ExportReport, CliError> {
    let registry = load_registry(&job.data)?;
    let (template, _) = load_template(&job.data)?;
    let template = Arc::new(template);
    let mut model = Model::new(template.clone(), &registry);
    let mut doc = default_document();
    for (id, value) in &job.params {
        if !registry.params().iter().any(|p| p.id == *id) {
            return Err(CliError::Data(format!("--set: no parameter `{id}` in the registry")));
        }
        let value: f64 = value.parse().map_err(|_| CliError::Data(format!("--set: `{value}` is not a number")))?;
        doc.params.insert(id.clone(), ParamValue::Scalar(value));
    }
    model.sync(&doc, &registry)?;
    let rest = RestShape::of(model.evaluator(), &model.input()).map_err(|e| CliError::Data(format!("export: {e}")))?;
    let options = ExportOptions { name: job.name.clone(), expressions: job.expressions, skin: *model.skin() };
    let (gltf, report) =
        character_gltf(&template, &rest, &options).map_err(|e| CliError::Data(format!("export: {e}")))?;
    let glb = gltf.to_glb().map_err(|e| CliError::Data(format!("export: {e}")))?;
    if let Some(dir) = job.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| CliError::io(dir, e))?;
    }
    std::fs::write(&job.out, glb).map_err(|e| CliError::io(&job.out, e))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(list: &[&str]) -> Result<ExportJob, CliError> {
        let args: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
        ExportJob::from_args(&Parsed::parse(&EXPORT, &args).map_err(CliError::Usage)?)
    }

    #[test]
    fn exp_p1_export_arguments_map_to_the_job() {
        let d = job(&[]).unwrap();
        assert_eq!((d.out.to_str(), d.expressions, d.name.as_str()), (Some("character.glb"), true, "character"));
        let j = job(&["--out", "a/b.glb", "--set", "identity.sex=1", "--expressions", "off", "--name", "bob"]).unwrap();
        assert_eq!(j.params, vec![("identity.sex".to_owned(), "1".to_owned())]);
        assert!(!j.expressions);
        for bad in [&["--expressions", "maybe"][..], &["--set", "x"], &["--name", " "]] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }

    /// The command writes a glTF binary of the repository's template that reads back.
    #[test]
    fn exp_p1_export_writes_a_glb_that_reads_back() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/export");
        let out = dir.join("character.glb");
        let mut j = job(&["--set", "identity.sex=1"]).unwrap();
        j.out.clone_from(&out);
        j.data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let report = run(&j).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let back = humanart_io::gltf::read_self_contained(&bytes, &humanart_io::Limits::DEFAULT).unwrap();
        assert_eq!(back.document.meshes[0].primitives.len(), report.primitives);
        assert_eq!(back.document.skins[0].joints.len(), report.joints);
        assert!(report.targets > 50);
        let bad = ExportJob { params: vec![("no.such".into(), "1".into())], ..j };
        assert!(matches!(run(&bad), Err(CliError::Data(_))));
    }
}
