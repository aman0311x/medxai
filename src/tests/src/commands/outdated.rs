use anyhow::Result;
use clap::Args;
use crate::config::DjuxProject;
use crate::registry::Registry;
use colored::*;

#[derive(Args, Debug)]
pub struct OutdatedArgs {
    // Puedes añadir opciones como --json o --verbose
}

pub fn run(args: OutdatedArgs) -> Result<()> {
    // 1. Cargar el archivo djux.project.json
    let project = DjuxProject::load()?;

    // 2. Obtener el registro de aplicaciones (desde una URL o archivo local)
    let registry = Registry::fetch()?;

    // 3. Comparar versiones
    let outdated_apps: Vec<_> = project
        .apps
        .iter()
        .filter_map(|app| {
            if let Some(latest_version) = registry.get_version(&app.name) {
                if latest_version > app.version {
                    return Some((app, latest_version));
                }
            }
            None
        })
        .collect();

    // 4. Imprimir una tabla clara
    if outdated_apps.is_empty() {
        println!("{}", "All apps are up to date.".green());
        return Ok(());
    }

    println!("{}", "Outdated apps:".yellow());
    for (app, latest) in outdated_apps {
        println!(
            "  {} {} -> {}",
            app.name.cyan(),
            app.version.to_string().red(),
            latest.to_string().green()
        );
    }

    Ok(())
}