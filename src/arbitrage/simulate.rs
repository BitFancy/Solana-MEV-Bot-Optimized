use std::collections::HashMap;

use log::info;
use log::error;
use rust_socketio::asynchronous::Client;

use crate::markets::meteora::simulate_route_meteora;
use crate::markets::{orca_whirpools::simulate_route_orca_whirpools, raydium::simulate_route_raydium, types::{DexLabel, Market}};
use super::types::{SwapPath, SwapRouteSimulation, TokenInfos};

pub async fn simulate_path(simulation_amount: u64, path: SwapPath, markets: Vec<Market>, tokens_infos: HashMap<String, TokenInfos>, mut route_simulation: HashMap<Vec<u32>, Vec<SwapRouteSimulation>>) -> (HashMap<Vec<u32>, Vec<SwapRouteSimulation>>, Vec<SwapRouteSimulation>, f64) {
    println!("🚕🚕🚕🚕  NEW PATH  🚕🚕🚕🚕");
    println!("Nb. Hops : {}", path.hops);
    let decimals = 9;
    let mut amount_in = simulation_amount;
    let amount_begin= amount_in;

    let mut swap_simulation_result: Vec<SwapRouteSimulation> = Vec::new();
    
    for (i, route) in path.paths.iter().enumerate() {
        let market: Option<Market> = markets.iter().cloned().find(|market| market.id == route.pool_address);

        match path.hops {
            1 => {
                if i == 0 && route_simulation.contains_key(&vec![path.id_paths[i]]) {
                    let swap_sim = route_simulation.get(&vec![path.id_paths[i]]).unwrap();
                    amount_in = swap_sim[0].estimated_amount_out.as_str().parse().expect("Bad conversion String to f64");
                    println!("📌 NO SIMULATION Route Id: {}", swap_sim[0].id_route);
                    swap_simulation_result.push(swap_sim[0].clone());
                    continue;
                }
            }
            2 => {
                if i == 0 && route_simulation.contains_key(&vec![path.id_paths[i]]) {
                    let swap_sim = route_simulation.get(&vec![path.id_paths[i]]).unwrap();
                    amount_in = swap_sim[0].estimated_amount_out.as_str().parse().expect("Bad conversion String to f64");
                    println!("📌 NO SIMULATION Route 1 Id: {}", swap_sim[0].id_route);
                    swap_simulation_result.push(swap_sim[0].clone());
                    continue;
                }
                if i == 1 {
                    if route_simulation.contains_key(&vec![path.id_paths[i - 1], path.id_paths[i]]) {
                        let swap_sim = route_simulation.get(&vec![path.id_paths[i - 1], path.id_paths[i]]).unwrap();
                        amount_in = swap_sim[1].estimated_amount_out.as_str().parse().expect("Bad conversion String to f64");
                        println!("📌 NO SIMULATION Route 2 Id: {}", swap_sim[1].id_route);
                        swap_simulation_result.push(swap_sim[1].clone());
                        continue;
                    }
                }
            }
            _ => {
                println!("⛔ Invalid number of hops")
            }
            //...
        }
        match route.dex {
            DexLabel::ORCA => {
                println!(" ⚠️⚠️ ONE ORCA POOL ");
            },
            DexLabel::ORCA_WHIRLPOOLS => {
                println!("🏊 ORCA_WHIRLPOOLS - POOL");
                println!("Address: {:?}", route.pool_address);
                match simulate_route_orca_whirpools(true, amount_in, route.clone(), market.unwrap(), tokens_infos.clone()).await {
                    Ok(value) => {
                        let (amount_out, min_amount_out) = value;
                        // println!("Amount out: {}", amount_out);
                        
                        let swap_sim: SwapRouteSimulation = SwapRouteSimulation{
                            id_route: route.id.clone(),
                            pool_address: route.pool_address.clone(),
                            dex_label: DexLabel::ORCA_WHIRLPOOLS,
                            token_0to1: route.token_0to1,
                            token_in: route.tokenIn.clone(),
                            token_out: route.tokenOut.clone(),
                            amount_in: amount_in,
                            estimated_amount_out: amount_out.clone(),
                            estimated_min_amount_out: min_amount_out.clone(),
                        };
        
                        //1rst route
                        if i == 0 && !route_simulation.contains_key(&vec![path.id_paths[i]]) {
                            route_simulation.insert(vec![route.id], vec![swap_sim.clone()]);
                        }
        
                        //2nd route
                        if i == 1 && path.hops == 2 && !route_simulation.contains_key(&vec![path.id_paths[i - 1], path.id_paths[i]]){
                            let swap_sim_prev_route = route_simulation.get(&vec![path.id_paths[i - 1]]).unwrap();
                            route_simulation.insert(vec![path.id_paths[i - 1], path.id_paths[i]], vec![swap_sim_prev_route[0].clone() , swap_sim.clone()]);
                        }
        
                        swap_simulation_result.push(swap_sim.clone());
                        amount_in = amount_out.as_str().parse().expect("Bad conversion String to f64");
                    }
                    Err(value) => {
                        // println!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("ORCA_WHIRLPOOLS POOL");
                        error!("Address: {:?}", route.pool_address);
                        // println!("❌ ERROR {:?}", value);
                        error!("ERROR {:?}", value);
                        println!("🔚 Skipped Path");
                        let empty_result: Vec<SwapRouteSimulation> = Vec::new();
                        return (route_simulation, empty_result, 0.0);
                    }
                }
            },
            DexLabel::RAYDIUM => {
                println!("🏊 RAYDIUM - POOL");
                println!("Address: {:?}", route.pool_address);
                match simulate_route_raydium(true, amount_in, route.clone(), market.unwrap(), tokens_infos.clone()).await {
                    Ok(value) => {
                        let (amount_out, min_amount_out) = value;
                        // println!("Amount out: {}", amount_out);
        
                        let swap_sim: SwapRouteSimulation = SwapRouteSimulation{
                            id_route: route.id.clone(),
                            pool_address: route.pool_address.clone(),
                            dex_label: DexLabel::RAYDIUM,
                            token_0to1: route.token_0to1,
                            token_in: route.tokenIn.clone(),
                            token_out: route.tokenOut.clone(),
                            amount_in: amount_in,
                            estimated_amount_out: amount_out.clone(),
                            estimated_min_amount_out: min_amount_out.clone(),
                        };
        
                        //1rst route
                        if i == 0 && !route_simulation.contains_key(&vec![path.id_paths[i]]) {
                            route_simulation.insert(vec![route.id], vec![swap_sim.clone()]);
                        }
                        //2nd route
                        if i == 1 && path.hops == 2 && !route_simulation.contains_key(&vec![path.id_paths[i - 1], path.id_paths[i]]){
                            let swap_sim_prev_route = route_simulation.get(&vec![path.id_paths[i - 1]]).unwrap();
                            route_simulation.insert(vec![path.id_paths[i - 1], path.id_paths[i]], vec![swap_sim_prev_route[0].clone() , swap_sim.clone()]);
                        }
                        
                        swap_simulation_result.push(swap_sim.clone());
                        amount_in = amount_out.as_str().parse().expect("Bad conversion String to f64");
                    }
                    Err(value) => {
                        // println!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("RAYDIUM POOL");
                        error!("Address: {:?}", route.pool_address);
                        // println!("❌ ERROR {:?}", value);
                        error!("ERROR {:?}", value);
                        println!("🔚 Skipped Path");
                        let empty_result: Vec<SwapRouteSimulation> = Vec::new();
                        return (route_simulation, empty_result, 0.0);
                    }
                }

            },
            DexLabel::RAYDIUM_CLMM => {
                println!(" ⚠️⚠️ ONE RAYDIUM_CLMM POOL ");
            },
            DexLabel::METEORA => {
                // println!(" ⚠️⚠️ ONE METEORA POOL ");
                println!("🏊 METEORA - POOL");
                println!("Address: {:?}", route.pool_address);
                match simulate_route_meteora(true, amount_in, route.clone(), market.unwrap(), tokens_infos.clone()).await {
                    Ok(value) => {
                        let (amount_out, min_amount_out) = value;
                        // println!("Amount out: {}", amount_out);
        
                        let swap_sim: SwapRouteSimulation = SwapRouteSimulation{
                            id_route: route.id.clone(),
                            pool_address: route.pool_address.clone(),
                            dex_label: DexLabel::METEORA,
                            token_0to1: route.token_0to1,
                            token_in: route.tokenIn.clone(),
                            token_out: route.tokenOut.clone(),
                            amount_in: amount_in,
                            estimated_amount_out: amount_out.clone(),
                            estimated_min_amount_out: min_amount_out.clone(),
                        };
        
                        //1rst route
                        if i == 0 && !route_simulation.contains_key(&vec![path.id_paths[i]]) {
                            route_simulation.insert(vec![route.id], vec![swap_sim.clone()]);
                        }
                        //2nd route
                        if i == 1 && path.hops == 2 && !route_simulation.contains_key(&vec![path.id_paths[i - 1], path.id_paths[i]]){
                            let swap_sim_prev_route = route_simulation.get(&vec![path.id_paths[i - 1]]).unwrap();
                            route_simulation.insert(vec![path.id_paths[i - 1], path.id_paths[i]], vec![swap_sim_prev_route[0].clone() , swap_sim.clone()]);
                        }
                        
                        swap_simulation_result.push(swap_sim.clone());
                        amount_in = amount_out.as_str().parse().expect("Bad conversion String to f64");
                    }
                    Err(value) => {
                        // println!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("METEORA POOL");
                        error!("Address: {:?}", route.pool_address);
                        // println!("❌ ERROR {:?}", value);
                        error!("ERROR {:?}", value);
                        println!("🔚 Skipped Path");
                        let empty_result: Vec<SwapRouteSimulation> = Vec::new();
                        return (route_simulation, empty_result, 0.0);
                    }
                }
            },
        }
    }
    info!("💵💵 Simulation of Swap Path [Id: {:?}] // Amount In: {} {} // Amount Out: {} {}", path.id_paths, amount_begin as f64 / 10_f64.powf(decimals as f64) , "SOL", amount_in as f64 / 10_f64.powf(decimals as f64), "SOL" );

    //If interesting path
    let difference = amount_in as f64 - amount_begin as f64;
    if difference > 0.0 {
        info!("💸💸💸💸💸💸💸💸💸💸 Path simulate {} {} positive difference", difference / 10_f64.powf(decimals as f64), "SOL");
    }

    return (route_simulation, swap_simulation_result, difference);
}

pub async fn simulate_path_precision(amount_input: u64, socket: Client, path: SwapPath, markets: Vec<Market>, tokens_infos: HashMap<String, TokenInfos>) -> (Vec<SwapRouteSimulation>, f64) {
    // println!("🚕🚕🚕🚕     NEW PRECISION PATH    🚕🚕🚕🚕");
    // println!("Nb. Hops : {}", path.hops);

    let decimals: u32 = 9;
    let amount_begin = amount_input;
    let mut amount_in = amount_input;

    let mut swap_simulation_result: Vec<SwapRouteSimulation> = Vec::new();
    
    for (i, route) in path.paths.iter().enumerate() {
        let market: Option<Market> = markets.iter().cloned().find(|market| market.id == route.pool_address);

        match route.dex {
            DexLabel::ORCA => {
                // println!(" ⚠️⚠️ ONE ORCA POOL ");
            },
            DexLabel::ORCA_WHIRLPOOLS => {
                // println!("ORCA_WHIRLPOOLS - POOL");
                // println!("Address: {:?}", route.pool_address);
                match simulate_route_orca_whirpools(false, amount_in, route.clone(), market.unwrap(), tokens_infos.clone()).await {
                    Ok(value) => {
                        let (amount_out, min_amount_out) = value;
                        // println!("Amount out: {}", amount_out);
                        
                        let swap_sim: SwapRouteSimulation = SwapRouteSimulation{
                            id_route: route.id.clone(),
                            pool_address: route.pool_address.clone(),
                            dex_label: DexLabel::ORCA_WHIRLPOOLS,
                            token_0to1: route.token_0to1,
                            token_in: route.tokenIn.clone(),
                            token_out: route.tokenOut.clone(),
                            amount_in: amount_in,
                            estimated_amount_out: amount_out.clone(),
                            estimated_min_amount_out: min_amount_out.clone(),
                        };
        
                        swap_simulation_result.push(swap_sim.clone());
                        amount_in = amount_out.as_str().parse().expect("Bad conversion String to f64");
                    }
                    Err(value) => {
                        // println!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("❌ PRECISION ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("ORCA_WHIRLPOOLS POOL");
                        error!("Address: {:?}", route.pool_address);
                        // println!("❌ ERROR {:?}", value);
                        error!("ERROR {:?}", value);
                        // println!("🔚 Skipped Path");
                        let empty_result: Vec<SwapRouteSimulation> = Vec::new();
                        return (empty_result, 0.0);
                    }
                }
            },
            DexLabel::RAYDIUM => {
                // println!("RAYDIUM - POOL");
                // println!("Address: {:?}", route.pool_address);
                match simulate_route_raydium(false, amount_in, route.clone(), market.unwrap(), tokens_infos.clone()).await {
                    Ok(value) => {
                        let (amount_out, min_amount_out) = value;
                        // println!("Amount out: {}", amount_out);
        
                        let swap_sim: SwapRouteSimulation = SwapRouteSimulation{
                            id_route: route.id.clone(),
                            pool_address: route.pool_address.clone(),
                            dex_label: DexLabel::RAYDIUM,
                            token_0to1: route.token_0to1,
                            token_in: route.tokenIn.clone(),
                            token_out: route.tokenOut.clone(),
                            amount_in: amount_in,
                            estimated_amount_out: amount_out.clone(),
                            estimated_min_amount_out: min_amount_out.clone(),
                        };
                        
                        swap_simulation_result.push(swap_sim.clone());
                        amount_in = amount_out.as_str().parse().expect("Bad conversion String to f64");
                    }
                    Err(value) => {
                        // println!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("❌ PRECISION ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("RAYDIUM POOL");
                        error!("Address: {:?}", route.pool_address);
                        // println!("❌ ERROR {:?}", value);
                        error!("ERROR {:?}", value);
                        println!("🔚 Skipped Path");
                        let empty_result: Vec<SwapRouteSimulation> = Vec::new();
                        return (empty_result, 0.0);
                    }
                }

            },
            DexLabel::RAYDIUM_CLMM => {
                // println!(" ⚠️⚠️ ONE RAYDIUM_CLMM POOL ");
            },
            DexLabel::METEORA => {
                // println!(" ⚠️⚠️ ONE METEORA POOL ");
                // println!("METEORA - POOL");
                // println!("Address: {:?}", route.pool_address);
                match simulate_route_meteora(false, amount_in, route.clone(), market.unwrap(), tokens_infos.clone()).await {
                    Ok(value) => {
                        let (amount_out, min_amount_out) = value;
                        // println!("Amount out: {}", amount_out);
        
                        let swap_sim: SwapRouteSimulation = SwapRouteSimulation{
                            id_route: route.id.clone(),
                            pool_address: route.pool_address.clone(),
                            dex_label: DexLabel::RAYDIUM,
                            token_0to1: route.token_0to1,
                            token_in: route.tokenIn.clone(),
                            token_out: route.tokenOut.clone(),
                            amount_in: amount_in,
                            estimated_amount_out: amount_out.clone(),
                            estimated_min_amount_out: min_amount_out.clone(),
                        };
                        
                        swap_simulation_result.push(swap_sim.clone());
                        amount_in = amount_out.as_str().parse().expect("Bad conversion String to f64");
                    }
                    Err(value) => {
                        // println!("❌ ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("❌ PRECISION ERROR HANDLED for route: {:?}", path.id_paths);
                        error!("METEORA POOL");
                        error!("Address: {:?}", route.pool_address);
                        // println!("❌ ERROR {:?}", value);
                        error!("ERROR {:?}", value);
                        // println!("🔚 Skipped Path");
                        let empty_result: Vec<SwapRouteSimulation> = Vec::new();
                        return (empty_result, 0.0);
                    }
                }
            },
        }
    }
    
    // info!("🔎🔎 Swap path Id: {:?}", path.id_paths);
    info!("🔎🔎💵💵 Precision Simulation: Amount In: {} {} // Amount Out: {} {}", amount_begin as f64 / 10_f64.powf(decimals as f64) , "SOL", amount_in as f64 / 10_f64.powf(decimals as f64), "SOL" );
    let difference = amount_in as f64 - amount_begin as f64;
    info!("🔎🔎 Path simulate {} {} difference", difference / 10_f64.powf(decimals as f64), "SOL");

    return (swap_simulation_result, difference);
}