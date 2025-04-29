use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path;

use anyhow::Result;
use futures::FutureExt;
use log::info;
use solana_sdk::pubkey::Pubkey;
use tokio::task::JoinSet;
use solana_client::rpc_client::RpcClient;
use MEV_Bot_Solana::arbitrage::strategies::{optimism_tx_strategy, run_arbitrage_strategy, sorted_interesting_path_strategy};
use MEV_Bot_Solana::common::database::insert_vec_swap_path_selected_collection;
use MEV_Bot_Solana::common::types::InputVec;
use MEV_Bot_Solana::markets::pools::load_all_pools;
use MEV_Bot_Solana::transactions::create_transaction::{create_ata_extendlut_transaction, ChainType, SendOrSimulate};
use MEV_Bot_Solana::{common::constants::Env, transactions::create_transaction::create_and_send_swap_transaction};
use MEV_Bot_Solana::common::utils::{from_str, get_tokens_infos, setup_logger};
use MEV_Bot_Solana::arbitrage::types::{SwapPathResult, SwapPathSelected, SwapRouteSimulation, TokenInArb, TokenInfos, VecSwapPathSelected};
use rust_socketio::{Payload, asynchronous::{Client, ClientBuilder},};


use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use mongodb::bson::doc;
use mongodb::{Client as MongoDbCLient, options::ClientOptions};

// use MEV_Bot_Solana::common::pools::{load_all_pools, Pool};

#[tokio::main]
async fn main() -> Result<()> {

    //Options
    let simulation_amount = 3500000000; //3.5 SOL
    // let simulation_amount = 1000000000; //1 SOL
    // let simulation_amount = 2000000000; //1 SOL

    let massive_strategie: bool = true;
    let best_strategie: bool = true;
    let optimism_strategie: bool = true;

    //massive_strategie options
    let fetch_new_pools = false;
            // Restrict USDC/SOL pools to 2 markets
    let restrict_sol_usdc = true;

    //best_strategie options
    // let mut path_best_strategie: String = format!("best_paths_selected/SOL-SOLLY.json");
    let mut path_best_strategie: String = format!("best_paths_selected/ultra_strategies/0-SOL-SOLLY-1-SOL-SPIKE-2-SOL-AMC-GME.json");
    
    
    //Optism tx to send
    let optimism_path: String = "optimism_transactions/11-6-2024-SOL-SOLLY-SOL-0.json".to_string();

    // //Send message to Rust execution program
    // let mut stream = TcpStream::connect("127.0.0.1:8080").await?;

    // let message = optimism_path.as_bytes();
    // stream.write_all(message).await?;
    // info!("🛜  Sent: {} tx to executor", String::from_utf8_lossy(message));

    let mut inputs_vec = vec![
        InputVec{
            tokens_to_arb: vec![
                TokenInArb{address: String::from("So11111111111111111111111111111111111111112"), symbol: String::from("SOL")}, // Base token here
                TokenInArb{address: String::from("4Cnk9EPnW5ixfLZatCPJjDB1PUtcRpVVgTQukm9epump"), symbol: String::from("DADDY-ANSEM")},
 
            ],
            include_1hop: true,
            include_2hop: true,
            numbers_of_best_paths: 4,
            // When we have more than 3 tokens it's better to desactivate caused by timeout on multiples getProgramAccounts calls
            get_fresh_pools_bool: false
        },
        InputVec{
            tokens_to_arb: vec![
                TokenInArb{address: String::from("So11111111111111111111111111111111111111112"), symbol: String::from("SOL")}, // Base token here
                TokenInArb{address: String::from("2J5uSgqgarWoh7QDBmHSDA3d7UbfBKDZsdy1ypTSpump"), symbol: String::from("DADDY-TATE")},

            ],
            include_1hop: true,
            include_2hop: true,
            numbers_of_best_paths: 4,
            // When we have more than 3 tokens it's better to desactivate caused by timeout on multiples getProgramAccounts calls
            get_fresh_pools_bool: false
        },
        InputVec{
            tokens_to_arb: vec![
                TokenInArb{address: String::from("So11111111111111111111111111111111111111112"), symbol: String::from("SOL")}, // Base token here
                TokenInArb{address: String::from("BX9yEgW8WkoWV8SvqTMMCynkQWreRTJ9ZS81dRXYnnR9"), symbol: String::from("SPIKE")},

            ],
            include_1hop: true,
            include_2hop: true,
            numbers_of_best_paths: 2,
            // When we have more than 3 tokens it's better to desactivate caused by timeout on multiples getProgramAccounts calls
            get_fresh_pools_bool: false
        },
        //////////////
        //////////////
        //////////////
        //////////////
        //////////////
        //////////////
        InputVec{
            tokens_to_arb: vec![
                TokenInArb{address: String::from("So11111111111111111111111111111111111111112"), symbol: String::from("SOL")}, // Base token here
                TokenInArb{address: String::from("9jaZhJM6nMHTo4hY9DGabQ1HNuUWhJtm7js1fmKMVpkN"), symbol: String::from("AMC")},
                TokenInArb{address: String::from("8wXtPeU6557ETkp9WHFY1n1EcU6NxDvbAggHGsMYiHsB"), symbol: String::from("GME")},
                // TokenInArb{address: String::from("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"), symbol: String::from("USDC")},
                // TokenInArb{address: String::from("5BKTP1cWao5dhr8tkKcfPW9mWkKtuheMEAU6nih2jSX"), symbol: String::from("NoHat")},
            ],
            include_1hop: true,
            include_2hop: true,
            numbers_of_best_paths: 4,
            // When we have more than 3 tokens it's better to desactivate caused by timeout on multiples getProgramAccounts calls
            get_fresh_pools_bool: false
        },
        // InputVec{
        //     tokens_to_arb: vec![
        //         TokenInArb{address: String::from("So11111111111111111111111111111111111111112"), symbol: String::from("SOL")}, // Base token here
        //         TokenInArb{address: String::from("8NH3AfwkizHmbVd83SSxc2YbsFmFL4m2BeepvL6upump"), symbol: String::from("TOPG")},
        //     ],
        //     include_1hop: true,
        //     include_2hop: true,
        //     numbers_of_best_paths: 2,
        //     get_fresh_pools_bool: false
        // },
    ];

    dotenv::dotenv().ok();
    setup_logger().unwrap();

    info!("Starting MEV_Bot_Solana");
    info!("⚠️⚠️ New fresh pools fetched on METEORA and RAYDIUM are excluded because a lot of time there have very low liquidity, potentially can be used on subscribe log strategy");
    info!("⚠️⚠️ Liquidity is fetch to API and can be outdated on Radyium Pool");

    let mut set: JoinSet<()> = JoinSet::new();
    
    // // The first token is the base token (here SOL)
    let tokens_to_arb: Vec<TokenInArb> = inputs_vec.clone().into_iter().flat_map(|input| input.tokens_to_arb).collect();

    info!("Open Socket IO channel...");
    let env = Env::new();
    
    let callback = |payload: Payload, socket: Client| {
        async move {
            match payload {
                Payload::String(data) => println!("Received: {}", data),
                Payload::Binary(bin_data) => println!("Received bytes: {:#?}", bin_data),
                Payload::Text(data) => println!("Received Text: {:?}", data),
            }
        }
        .boxed()
    };
    
    let mut socket = ClientBuilder::new("http://localhost:3000")
        .namespace("/")
        .on("connection", callback)
        .on("error", |err, _| {
            async move { eprintln!("Error: {:#?}", err) }.boxed()
        })
        .on("orca_quote", callback)
        .on("orca_quote_res", callback)
        .connect()
        .await
        .expect("Connection failed");


    if massive_strategie {
        info!("🏊 Launch pools fetching infos...");
        let dexs = load_all_pools(fetch_new_pools).await;
        info!("🏊 {} Dexs are loaded", dexs.len());
        
        
        info!("🪙🪙 Tokens Infos: {:?}", tokens_to_arb);
        info!("📈 Launch arbitrage process...");
        let mut vec_best_paths:Vec<String> = Vec::new();
        for input_iter in inputs_vec.clone() {
            let tokens_infos: HashMap<String, TokenInfos> = get_tokens_infos(input_iter.tokens_to_arb.clone()).await;

            let result = run_arbitrage_strategy(simulation_amount, input_iter.get_fresh_pools_bool, restrict_sol_usdc, input_iter.include_1hop, input_iter.include_2hop, input_iter.numbers_of_best_paths, dexs.clone(), input_iter.tokens_to_arb.clone(), tokens_infos.clone()).await;
            let (path_for_best_strategie, swap_path_selected) = result.unwrap();
            vec_best_paths.push(path_for_best_strategie);
        }
        if inputs_vec.clone().len() > 1 {
            let mut vec_to_ultra_strat: Vec<SwapPathSelected> = Vec::new();
            let mut ultra_strat_name: String = format!("");
            for (index, iter_path) in vec_best_paths.iter().enumerate() {
                let name_raw: Vec<&str> = iter_path.split('/').collect();
                let name: Vec<&str> = name_raw[1].split('.').collect();
                if index == 0 {
                    ultra_strat_name = format!("{}-{}", index, name[0]);
                } else {
                    ultra_strat_name = format!("{}-{}-{}", ultra_strat_name, index, name[0]);
                }

                let file_read = OpenOptions::new().read(true).write(true).open(iter_path)?;
                let mut paths_vec: VecSwapPathSelected = serde_json::from_reader(&file_read).unwrap();
                for sp_iter in paths_vec.value {
                    vec_to_ultra_strat.push(sp_iter);
                }
            }
            let mut path = format!("best_paths_selected/ultra_strategies/{}.json", ultra_strat_name);
            File::create(path.clone());
        
            let file = OpenOptions::new().read(true).write(true).open(path.clone())?;
            let mut writer = BufWriter::new(&file);
        
            let mut content = VecSwapPathSelected{value: vec_to_ultra_strat.clone()};
            writer.write_all(serde_json::to_string(&content)?.as_bytes())?;
            writer.flush()?;
            info!("Data written to '{}' successfully.", path);

            insert_vec_swap_path_selected_collection("ultra_strategies", content).await;

            path_best_strategie = path;
        }

        if best_strategie {
            let tokens_infos: HashMap<String, TokenInfos> = get_tokens_infos(tokens_to_arb.clone()).await;

            let _ = sorted_interesting_path_strategy(simulation_amount, path_best_strategie.clone(), tokens_to_arb.clone(), tokens_infos.clone()).await;
        }
    }
    
    if best_strategie {
        let tokens_infos: HashMap<String, TokenInfos> = get_tokens_infos(tokens_to_arb.clone()).await;

        let _ = sorted_interesting_path_strategy(simulation_amount, path_best_strategie.clone(), tokens_to_arb.clone(), tokens_infos.clone()).await;
    }
    
    if optimism_strategie {
        let _ = optimism_tx_strategy(optimism_path);
    }
    
    while let Some(res) = set.join_next().await {
        info!("{:?}", res);
    }

    println!("End");
    Ok(())
}