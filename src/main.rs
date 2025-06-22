use solana_vntr_sniper::{
    common::{config::Config, constants::RUN_MSG},
    engine::monitor::new_token_trader_pumpfun,
};

#[tokio::main]
async fn main() {
    /* Initial Settings */
    let config = Config::new().await;
    let config = config.lock().await;

    /* Running Bot */
    let run_msg = RUN_MSG;
    println!("{}", run_msg);

    let _ = new_token_trader_pumpfun(
        config.yellowstone_grpc_http.clone(),
        config.yellowstone_grpc_token.clone(),
        config.app_state.clone(),
        config.swap_config.clone(),
        config.blacklist.clone(),
        config.time_exceed,
        config.counter_limit as u64,
        config.min_dev_buy as u64,
        config.max_dev_buy as u64,
    )
    .await;
}
