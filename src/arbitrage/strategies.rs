use std::{collections::HashMap, fs::{File, OpenOptions}, thread::sleep, time::{self, SystemTime}};
use borsh::error;
use chrono::{Datelike, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::enumerate;
use mongodb::bson::doc;
use rust_socketio::{asynchronous::{Client}};
use solana_sdk::pubkey::Pubkey;
use std::io::{BufWriter, Write};
use crate::{arbitrage::{
    calc_arb::{calculate_arb, get_markets_arb}, simulate::simulate_path, streams::get_fresh_accounts_states, types::{SwapPathResult, SwapPathSelected, SwapRouteSimulation, VecSwapPathResult, VecSwapPathSelected}
}, common::{database::{insert_vec_swap_path_selected_collection, insert_swap_path_result_collection}, utils::{from_str, write_file_swap_path_result}}, transactions::create_transaction::{self, create_and_send_swap_transaction, create_ata_extendlut_transaction, ChainType, SendOrSimulate}};
use crate::markets::types::{Dex,Market};
use super::{simulate::simulate_path_precision, types::{SwapPath, TokenInArb, TokenInfos}};
use log::{debug, error, info};
use anyhow::Result;

use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub async fn run_arbitrage_strategy(simulation_amount: u64, get_fresh_pools_bool: bool, restrict_sol_usdc: bool, include_1hop: bool, include_2hop: bool, numbers_of_best_paths: usize, dexs: Vec<Dex>, tokens: Vec<TokenInArb>, tokens_infos: HashMap<String, TokenInfos>) -> Result<(String, VecSwapPathSelected)> {
    info!("👀 Run Arbitrage Strategies...");

    let markets_arb = get_markets_arb(get_fresh_pools_bool, restrict_sol_usdc, dexs, tokens.clone()).await;

    // println!("DEBUG {:?}", fresh_markets_arb);
    // debug!("DEBUG {:?}", markets_arb.get(&"3s3CzbFzkqLvXYA93M3uHCes2nc4SiuZ11emtpDJwCht".to_string()));
    // debug!("DEBUG {:?}", fresh_markets_arb.get(&"65shmpuYmxx5p7ggNCZbyrGLCXVqbBR1ZD5aAocRBUNG".to_string()));

    // Sort markets with low liquidity
    let (sorted_markets_arb, all_paths) = calculate_arb(include_1hop, include_2hop, markets_arb.clone(), tokens.clone());

    //Get fresh account state
    let fresh_markets_arb = get_fresh_accounts_states(sorted_markets_arb.clone()).await;  
    
    // We keep route simulation result for RPC optimization
    let mut route_simulation: HashMap<Vec<u32>, Vec<SwapRouteSimulation>> = HashMap::new();
    let mut swap_paths_results: VecSwapPathResult = VecSwapPathResult{result: Vec::new()};

    let mut counter_failed_paths = 0;
    let mut counter_positive_paths = 0;
    let mut error_paths: HashMap<Vec<u32>, u8> = HashMap::new();
    
    //Progress bar
    let bar = ProgressBar::new(all_paths.len() as u64);
    bar.set_style(ProgressStyle::with_template("[{elapsed}] [{bar:140.cyan/blue}] ✅ {pos:>3}/{len:3} {msg}")
    .unwrap()
    .progress_chars("##-"));
    bar.set_message(format!("❌ Failed routes: {}/{} 💸 Positive routes: {}/{}", counter_failed_paths, bar.position(), counter_positive_paths, bar.position()));

    let mut best_paths_for_strat: Vec<SwapPathSelected> = Vec::new();
    // for i in 0..numbers_of_best_paths {
    //     best_paths_for_strat.push((-1000000000.0, SwapPath{ hops: 0, paths: Vec::new(), id_paths: Vec::new() }));
    // }
    //Begin simulate all paths
    let mut return_path = "".to_string();
    let mut counter_sp_result = 0;

    for (i, path) in all_paths.iter().enumerate() {     //Add this to limit iterations: .take(100)
        // println!("👀 Swap paths: {:?}", path);

        // Verify error in previous paths to see if this path is interesting
        let key = vec![path.id_paths[0], path.id_paths[1]];
        let counter_opt = error_paths.get(&key.clone());
        match counter_opt {
            None => {},
            Some(value) => {
                if value >= &3 {
                    error!("🔴⏭️  Skip the {:?} path because previous errors", path.id_paths);
                    bar.inc(1);
                    counter_failed_paths += 1;
                    bar.set_message(format!("❌ Failed routes: {}/{} 💸 Positive routes: {}/{}", counter_failed_paths, bar.position(), counter_positive_paths, bar.position()));
                    continue;
                }
            }
        }


        // Get Pubkeys of the concerned markets
        let pubkeys: Vec<String> = path.paths.clone().iter().map(|route| route.clone().pool_address).collect();
        let markets: Vec<Market> = pubkeys.iter().filter_map(|key| fresh_markets_arb.get(key)).cloned().collect();

        let (new_route_simulation, swap_simulation_result, result_difference) = simulate_path(simulation_amount, path.clone(), markets.clone(), tokens_infos.clone(), route_simulation.clone()).await;
        
        //If no error in swap path
        if swap_simulation_result.len() >= path.hops as usize {
            // tokens.iter().map(|token| &token.symbol).cloned().collect::<Vec<String>>().join("-");

            let mut tokens_path = swap_simulation_result.iter().map(|swap_sim| tokens_infos.get(&swap_sim.token_in).unwrap().symbol.clone()).collect::<Vec<String>>().join("-");
            tokens_path = format!("{}-{}",tokens_path, tokens[0].symbol.clone());

            let sp_result: SwapPathResult = SwapPathResult{ 
                path_id: i as u32, 
                hops: path.hops,
                tokens_path: tokens_path.clone(), 
                route_simulations: swap_simulation_result.clone(), 
                token_in: tokens[0].address.clone(), 
                token_in_symbol: tokens[0].symbol.clone(), 
                token_out: tokens[0].address.clone(), 
                token_out_symbol: tokens[0].symbol.clone(), 
                amount_in: swap_simulation_result[0].amount_in.clone(), 
                estimated_amount_out: swap_simulation_result[swap_simulation_result.len() - 1].estimated_amount_out.clone(), 
                estimated_min_amount_out: swap_simulation_result[swap_simulation_result.len() - 1].estimated_min_amount_out.clone(), 
                result: result_difference
            };
            swap_paths_results.result.push(sp_result.clone());

            if result_difference > 20000000.0 {
                println!("💸💸💸💸💸💸💸💸💸 Begin Execute the tx 💸💸💸💸💸💸💸💸💸");
                info!("💸💸💸💸💸💸💸💸💸 Send transaction execution... 💸💸💸💸💸💸💸💸💸");
                
                let now = Utc::now();
                let date = format!("{}-{}-{}", now.day(), now.month(), now.year());

                let path = format!("optimism_transactions/{}-{}-{}.json", date, tokens_path.clone(), counter_sp_result);
                let _ = insert_swap_path_result_collection("optimism_transactions", sp_result.clone()).await;  
                let _ = write_file_swap_path_result(path.clone(), sp_result);
                counter_sp_result += 1;
                
                //Send message to Rust execution program
                let mut stream = TcpStream::connect("127.0.0.1:8080").await?;

                let message = path.as_bytes();
                stream.write_all(message).await?;
                info!("🛜  Sent: {} tx to executor", String::from_utf8_lossy(message));
                // let mut buffer = [0; 512];
                // let n = stream.read(&mut buffer).await?;
                // info!("Received: {}", String::from_utf8_lossy(&buffer[0..n]));
            }

            // Reset errors if one path is good to only skip paths on 3 consecutives errors
            let key = vec![path.id_paths[0], path.id_paths[1]];
            error_paths.insert(key, 0);

            //Custom Queue FIFO for best results
            if best_paths_for_strat.len() < numbers_of_best_paths {
                best_paths_for_strat.push(SwapPathSelected{result: result_difference, path: path.clone(), markets: markets});
                if best_paths_for_strat.len() == numbers_of_best_paths {
                    best_paths_for_strat.sort_by(|a, b| b.result.partial_cmp(&a.result).unwrap());
                }
            } else if result_difference > best_paths_for_strat[best_paths_for_strat.len() - 1].result {
                for (index, path_in_vec) in best_paths_for_strat.clone().iter().enumerate() {
                    if result_difference < path_in_vec.result {
                        continue;
                    } else {
                        best_paths_for_strat[index] = SwapPathSelected{result: result_difference, path: path.clone(), markets: markets};
                        break;
                    }
                }
                // best_paths_for_strat.remove(0);
                // best_paths_for_strat.push((result_difference, path.clone()));
            }

            if i % 10 == 0 {
                println!("best_paths_for_strat {:#?}", best_paths_for_strat.iter().map(|iter| iter.result).collect::<Vec<f64>>());
            }
            // Update positive routes
            if result_difference > 0.0 {
                counter_positive_paths += 1;
                bar.set_message(format!("❌ Failed routes: {}/{} 💸 Positive routes: {}/{}", counter_failed_paths, bar.position(), counter_positive_paths, bar.position()));

                // precision_strategy(socket.clone(), path.clone(), markets, tokens.clone(), tokens_infos.clone()).await;
            }
        } else {
            counter_failed_paths += 1;

            //Code to avoid simulations of all paths on min. 3 likely consecutive failed paths, for 2 hops here
            if swap_simulation_result.len() == 0 {
                let key = vec![path.id_paths[0], path.id_paths[1]];
                let counter_opt = error_paths.get(&key.clone());
                match counter_opt {
                    None => {
                        error_paths.insert(key, 1);
                    }
                    Some(value) => {
                        error_paths.insert(key, value + 1);
                    }
                }
            }
        }

        route_simulation = new_route_simulation;

        bar.inc(1);
        bar.set_message(format!("❌ Failed routes: {}/{} 💸 Positive routes: {}/{}", counter_failed_paths, bar.position(), counter_positive_paths, bar.position()));

        if (i != 0 && i % 300 == 0) || i == all_paths.len() - 1 {
            let file_number = i / 300;
            let symbols = tokens.iter().map(|token| &token.symbol).cloned().collect::<Vec<String>>().join("-");
            let mut file = File::create(format!("results\\result_{}_{}.json", file_number, symbols)).unwrap();
            match serde_json::to_writer_pretty(&mut file, &swap_paths_results) {
                Ok(value) => {
                    info!("🥇🥇 Results writed!");
                    swap_paths_results = VecSwapPathResult{result: Vec::new()};
                }
                Err(value) => {
                    error!("Results not writed well: {:?}", value);
                }
            };
        }
        // println!("🥇🥇 Results of swaps paths");
        // for sp in swap_paths_results.result {
        //     info!("🔎 Path Id: {} // {} hop(s)", sp.path_id, sp.hops);
        //     info!("amount_in: {} {}", sp.amount_in, sp.token_in_symbol);
        //     info!("estimated_amount_out: {} {}", sp.estimated_amount_out, sp.token_out_symbol);
        // }
        
    }
    
    let mut tokens_list = "".to_string();
    for (index, token) in tokens.iter().enumerate() {
        if index == 0 {
            tokens_list = format!("{}", tokens[index].symbol.clone());
        } else {
            tokens_list = format!("{}-{}", tokens_list, tokens[index].symbol.clone());
        }
    }

    
    let mut path = format!("best_paths_selected/{}.json", tokens_list);
    File::create(path.clone());
    
    let file = OpenOptions::new().read(true).write(true).open(path.clone())?;
    let mut writer = BufWriter::new(&file);
    
    let mut content = VecSwapPathSelected{value: best_paths_for_strat.clone()};
    writer.write_all(serde_json::to_string(&content)?.as_bytes())?;
    writer.flush()?;
    info!("Data written to '{}' successfully.", path);
    
    insert_vec_swap_path_selected_collection("best_paths_selected", content.clone()).await;

    return_path = path;
    bar.finish();
    return Ok((return_path, VecSwapPathSelected{ value: best_paths_for_strat}));
}

pub async fn precision_strategy(socket: Client, path: SwapPath, markets: Vec<Market>, tokens: Vec<TokenInArb>, tokens_infos: HashMap<String, TokenInfos>) {

    info!("🔎🔎 Run a Precision SImulation on Path Id: {:?}", path.id_paths);

    let mut swap_paths_results: VecSwapPathResult = VecSwapPathResult{result: Vec::new()};

    let decimals = 9;
    let amounts_simulations = vec![
        5 * 10_u64.pow(decimals - 1),
        1 * 10_u64.pow(decimals),
        // 2 * 10_u64.pow(decimals),
        // 3 * 10_u64.pow(decimals),
        5 * 10_u64.pow(decimals),
        10 * 10_u64.pow(decimals),
        20 * 10_u64.pow(decimals)
    ];
    
    let mut result_amt = 0.0;
    let mut sp_to_tx: Option<SwapPathResult> = None;

    for (index, amount_in) in amounts_simulations.iter().enumerate() {
        let (swap_simulation_result, result_difference) = simulate_path_precision(amount_in.clone(), socket.clone(), path.clone(), markets.clone(), tokens_infos.clone()).await;

        if swap_simulation_result.len() >= path.hops as usize {
            let mut tokens_path = swap_simulation_result.iter().map(|swap_sim| tokens_infos.get(&swap_sim.token_in).unwrap().symbol.clone()).collect::<Vec<String>>().join("-");
            tokens_path = format!("{}-{}",tokens_path, tokens[0].symbol.clone());

            let sp_result: SwapPathResult = SwapPathResult{ 
                path_id: index as u32, 
                hops: path.hops, 
                tokens_path: tokens_path,
                route_simulations: swap_simulation_result.clone(), 
                token_in: tokens[0].address.clone(), 
                token_in_symbol: tokens[0].symbol.clone(), 
                token_out: tokens[0].address.clone(), 
                token_out_symbol: tokens[0].symbol.clone(), 
                amount_in: swap_simulation_result[0].amount_in.clone(), 
                estimated_amount_out: swap_simulation_result[swap_simulation_result.len() - 1].estimated_amount_out.clone(), 
                estimated_min_amount_out: swap_simulation_result[swap_simulation_result.len() - 1].estimated_min_amount_out.clone(), 
                result: result_difference
            };
            swap_paths_results.result.push(sp_result.clone());
            
            if result_difference > result_amt {
                result_amt = result_difference;
                println!("result_amt: {}", result_amt);
                sp_to_tx = Some(sp_result.clone());
            }
        }
    }
    if result_amt > 0.1 && sp_to_tx.is_some() {
        // let _ = create_and_send_swap_transaction(
        //     create_transaction::SendOrSimulate::Simulate, 
        //     create_transaction::ChainType::Mainnet, 
        //     sp_to_tx.unwrap()
        // ).await;
    }
}   

pub async fn sorted_interesting_path_strategy(simulation_amount: u64, path:String, tokens: Vec<TokenInArb>, tokens_infos: HashMap<String, TokenInfos>) -> Result<()>{

    let file_read = OpenOptions::new().read(true).write(true).open(path)?;
    let mut paths_vec: VecSwapPathSelected = serde_json::from_reader(&file_read).unwrap();
    let mut counter_sp_result = 0;

    let paths: Vec<SwapPathSelected> = paths_vec.value;
    let mut route_simulation: HashMap<Vec<u32>, Vec<SwapRouteSimulation>> = HashMap::new();
    let tokens_for_tx: Vec<Pubkey> = tokens.iter().map(|tk| from_str(&tk.address).unwrap()).collect();
    loop {
        for (index, path) in paths.iter().enumerate() {
            let (new_route_simulation, swap_simulation_result, result_difference) = simulate_path(simulation_amount, path.path.clone(), path.markets.clone(), tokens_infos.clone(), route_simulation.clone()).await;
            //If no error in swap path
            if swap_simulation_result.len() >= path.path.hops as usize {
                // tokens.iter().map(|token| &token.symbol).cloned().collect::<Vec<String>>().join("-");
        
                let mut tokens_path = swap_simulation_result.iter().map(|swap_sim| tokens_infos.get(&swap_sim.token_in).unwrap().symbol.clone()).collect::<Vec<String>>().join("-");
                tokens_path = format!("{}-{}",tokens_path, tokens[0].symbol.clone());
        
                let sp_result: SwapPathResult = SwapPathResult{ 
                    path_id: index as u32, 
                    hops: path.path.hops,
                    tokens_path: tokens_path.clone(), 
                    route_simulations: swap_simulation_result.clone(), 
                    token_in: tokens[0].address.clone(), 
                    token_in_symbol: tokens[0].symbol.clone(), 
                    token_out: tokens[0].address.clone(), 
                    token_out_symbol: tokens[0].symbol.clone(), 
                    amount_in: swap_simulation_result[0].amount_in.clone(), 
                    estimated_amount_out: swap_simulation_result[swap_simulation_result.len() - 1].estimated_amount_out.clone(), 
                    estimated_min_amount_out: swap_simulation_result[swap_simulation_result.len() - 1].estimated_min_amount_out.clone(), 
                    result: result_difference
                };
                
                if result_difference > 20000000.0 {
                    println!("💸💸💸💸💸💸💸💸💸 Begin Execute the tx 💸💸💸💸💸💸💸💸💸");
                    info!("💸💸💸💸💸💸💸💸💸 Send transaction execution... 💸💸💸💸💸💸💸💸💸");
                    // let _ = create_ata_extendlut_transaction(
                        //     ChainType::Mainnet,
                        //     SendOrSimulate::Send,
                    //     sp_result.clone(),
                    //     from_str("6nGymM5X1djYERKZtoZ3Yz3thChMVF6jVRDzhhcmxuee").unwrap(),
                    //     tokens_for_tx.clone()
                    // ).await;
                    // let _ = create_and_send_swap_transaction(
                    //     SendOrSimulate::Send,
                    //     ChainType::Mainnet, 
                    //     sp_result.clone()
                    // ).await;
                    
                    let now = Utc::now();
                    let date = format!("{}-{}-{}", now.day(), now.month(), now.year());

                    let path = format!("optimism_transactions/{}-{}-{}.json", date, tokens_path, counter_sp_result);
                    let _ = write_file_swap_path_result(path.clone(), sp_result);
                    counter_sp_result += 1;
                    
                    //Send message to Rust execution program
                    let mut stream = TcpStream::connect("127.0.0.1:8080").await?;
    
                    let message = path.as_bytes();
                    stream.write_all(message).await?;
                    info!("🛜  Sent: {} tx to executor", String::from_utf8_lossy(message));
                    // let mut buffer = [0; 512];
                    // let n = stream.read(&mut buffer).await?;
                    // info!("Received: {}", String::from_utf8_lossy(&buffer[0..n]));
                }
            }
            sleep(time::Duration::from_millis(200))
        }
    }
    // Ok(())

}

pub async fn optimism_tx_strategy(path:String) -> Result<()>{

    let file_read = OpenOptions::new().read(true).write(true).open(path)?;
    let mut spr: SwapPathResult = serde_json::from_reader(&file_read).unwrap();
    let mut counter_sp_result = 0;

    println!("💸💸💸💸💸💸💸💸💸 Begin Execute the tx 💸💸💸💸💸💸💸💸💸");
    // let _ = create_ata_extendlut_transaction(
    //     ChainType::Mainnet,
    //     SendOrSimulate::Send,
    //     sp_result.clone(),
    //     from_str("6nGymM5X1djYERKZtoZ3Yz3thChMVF6jVRDzhhcmxuee").unwrap(),
    //     tokens_for_tx.clone()
    // ).await;
    let _ = create_and_send_swap_transaction(
        SendOrSimulate::Send,
        ChainType::Mainnet, 
        spr.clone()
    ).await;

    Ok(())

}