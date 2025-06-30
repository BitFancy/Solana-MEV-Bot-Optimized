mod bot;
mod config;
mod constants;
mod dex;
mod pools;
mod refresh;
mod transaction;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use dotenv::dotenv;
use crate::config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global default subscriber");

    info!("Starting Solana Onchain Bot");

    // Initialize config singleton
    let _config = Config::new().await;
    // Use Config::get().await to access config anywhere

    bot::run_bot().await?;

    Ok(())
}
