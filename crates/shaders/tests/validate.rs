//! `[AC-WP05-1]` `[PLAT-04]` `[PERF-R5]`: every shader of the manifest compiles for every flag set
//! and every override combination it lists: first with naga alone (no GPU), then as pipelines on
//! the real device.

use humanart_gpu::{DeviceRequest, Gpu};
use humanart_shaders::manifest;
use humanart_shaders::validate::{self, Report};

/// The number of override combinations of every shader, summed.
fn expected_variants() -> usize {
    manifest::shaders().map(|s| s.flag_sets.len() * s.combination_count()).sum()
}

#[test]
fn ac_wp05_1_every_shader_and_combination_validates_with_naga() {
    let report = validate::validate_all_naga().unwrap_or_else(|e| panic!("{e}"));
    let shaders = manifest::shaders().count();
    assert_eq!(report.shaders, shaders);
    assert_eq!(report.variants, expected_variants());
    assert!(report.variants > shaders, "the override matrix is exercised");
    println!("naga: {report:?}");
}

#[test]
fn ac_wp05_1_every_pipeline_of_every_combination_builds_on_the_device() {
    let gpu = Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU");
    let report: Report = validate::validate_all_on_device(&gpu).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.shaders, manifest::shaders().count());
    assert_eq!(report.skipped_flag_sets, 0, "every flag set runs on this machine");
    assert_eq!(report.variants, expected_variants());
    let expected_pipelines: usize = manifest::shaders()
        .map(|s| {
            let compute = s.entry_points.iter().filter(|e| e.stage == manifest::Stage::Compute).count();
            let vertex = s.entry_points.iter().filter(|e| e.stage == manifest::Stage::Vertex).count();
            let fragment = s.entry_points.iter().filter(|e| e.stage == manifest::Stage::Fragment).count();
            (compute + vertex * fragment.max(1)) * s.combination_count() * s.flag_sets.len()
        })
        .sum();
    assert_eq!(report.pipelines, expected_pipelines);
    println!("device ({}): {report:?}", gpu.adapter_info().name);
}

#[test]
fn a_broken_variant_is_reported_by_the_device() {
    use manifest::{EntryPoint, Override, OverrideType, Purpose, ShaderDesc, Stage, Targets};
    // A manifest entry whose second combination has a workgroup size of zero. It reuses a real
    // file, so only the override list is wrong.
    let desc = ShaderDesc {
        path: "test/brdf.wgsl",
        summary: "broken on purpose",
        purpose: Purpose::Test,
        entry_points: &[
            EntryPoint { name: "furnace_rows", stage: Stage::Compute },
            EntryPoint { name: "dfg", stage: Stage::Compute },
            EntryPoint { name: "fresnel", stage: Stage::Compute },
        ],
        overrides: &[
            Override { name: "WORKGROUP_SIZE", ty: OverrideType::U32, values: &[64.0, 0.0] },
            Override { name: "SAMPLE_COUNT", ty: OverrideType::U32, values: &[16.0] },
        ],
        flag_sets: manifest::NO_FLAGS,
        targets: Targets::NONE,
    };
    let mut report = Report::default();
    let naga = validate::validate_naga(&desc, &mut report).unwrap_err();
    assert!(naga.to_string().contains("test/brdf.wgsl"), "{naga}");
    let gpu = Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU");
    let error = validate::validate_on_device(&gpu, &desc, &mut Report::default()).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("WORKGROUP_SIZE=0"), "{message}");
}
