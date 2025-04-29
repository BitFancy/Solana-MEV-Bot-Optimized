use anchor_spl::token::spl_token::error;
use itertools::Itertools;
use log::info;
use serde::{Deserialize, Serialize};
use solana_client::{connection_cache::ConnectionCache, rpc_client::RpcClient, rpc_config::{RpcSendTransactionConfig, RpcSimulateTransactionConfig}, send_and_confirm_transactions_in_parallel::{send_and_confirm_transactions_in_parallel, SendAndConfirmConfig}, tpu_client::{TpuClient, TpuClientConfig}};
use solana_sdk::{
    address_lookup_table::{instruction::{create_lookup_table, extend_lookup_table},
    state::AddressLookupTable, AddressLookupTableAccount}, commitment_config::{CommitmentConfig, CommitmentLevel}, compute_budget::ComputeBudgetInstruction, instruction::Instruction, message::{v0, VersionedMessage}, pubkey::Pubkey, signature::{read_keypair_file, Keypair, Signer}, sysvar::instructions, transaction::VersionedTransaction
};
use solana_transaction_status::UiTransactionEncoding;
use anchor_spl::token::spl_token;
use solana_sdk::{transaction::Transaction};
use anyhow::Result;
use log::error;
use spl_associated_token_account::{get_associated_token_address};
use spl_associated_token_account::instruction::create_associated_token_account;
use std::{fs::{File, OpenOptions}, io::{BufReader, Read}, path::Path, sync::Arc, thread::sleep, time::{Duration, Instant}};
use std::io::{BufWriter, Write};

use crate::{arbitrage::types::SwapPathResult, common::{constants::Env, utils::from_str}, markets::types::DexLabel, transactions::utils::{average, check_tx_status}};
use super::{meteoradlmm_swap::{construct_meteora_instructions, SwapParametersMeteora}, orca_whirpools_swap::{construct_orca_whirpools_instructions, SwapParametersOrcaWhirpools}, raydium_swap::{construct_raydium_instructions, SwapParametersRaydium}};

pub async fn create_and_send_swap_transaction(simulate_or_send: SendOrSimulate, chain: ChainType, transaction_infos: SwapPathResult) -> Result<()> {
    info!("🔄 Create swap transaction.... ");
    
    let env = Env::new();
    let rpc_url = if chain.clone() == ChainType::Mainnet { env.rpc_url_tx.clone() } else { env.devnet_rpc_url };
    let rpc_client: RpcClient = RpcClient::new(rpc_url);

    let payer: Keypair = read_keypair_file(env.payer_keypair_path.clone()).expect("Wallet keypair file not found");
    info!("💳 Wallet {:#?}", payer.pubkey());

    info!("🆔 Create/Send Swap instruction....");
    // Construct Swap instructions
    let compute_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(1_400_000);
    let compute_budget_instruction = vec![InstructionDetails{ instruction: compute_budget_ix, details: "Compute Budget Instruction".to_string(), market: None }];
    let priority_fees_ix = ComputeBudgetInstruction::set_compute_unit_price(0);
    let priority_fees_instruction = vec![InstructionDetails{ instruction: priority_fees_ix, details: "Set priority fees".to_string(), market: None }];


    let swaps_construct_instructions: Vec<InstructionDetails> = construct_transaction(transaction_infos).await;
    let mut swap_instructions: Vec<InstructionDetails> = vec![compute_budget_instruction, priority_fees_instruction, swaps_construct_instructions].concat();

    if swap_instructions.len() == 0 {
        error!("Error in create_transaction(), zero instructions");
        return Ok(());
    }
    
    // Keep the accounts which are not in previously crafted LUT tables
    let mut lut_addresses: Vec<Pubkey> = Vec::new();
    for (i, si) in swap_instructions.clone().iter().enumerate() {
        if let Some(market_addr) = si.market.as_ref().and_then(|m| Some(m.address)) {
            let (have_lut_address, lut_address) = get_lut_address_for_market(market_addr, false).unwrap();
            match have_lut_address {
                true => {
                    if !lut_addresses.contains(&lut_address.unwrap()) {
                        info!("LUT address {} pushed!", &lut_address.unwrap());
                        lut_addresses.push(lut_address.unwrap());
                    }
                    // Get the accounts we need for this swap with the lookup
                    // match si.market.as_ref().unwrap().dex_label {
                    //     DexLabel::METEORA => {
                    //         //Here we keep the bin arrays
                    //         let vec_accounts = &si.instruction.accounts;
                    //         let len = vec_accounts.len();
                    //         let new_vec = vec_accounts[len - 3..len].to_vec();
                    //         swap_instructions[i].instruction.accounts = new_vec
                    //     }
                    //     DexLabel::RAYDIUM => {
                    //         //Here nothing to keep
                    //         let new_vec = Vec::new();
                    //         swap_instructions[i].instruction.accounts = new_vec
                    //     }
                    //     DexLabel::ORCA_WHIRLPOOLS => {
                    //         //Here we keep the tick arrays
                    //         let vec_accounts = &si.instruction.accounts;
                    //         let len = vec_accounts.len();
                    //         let new_vec = vec_accounts[len - 4..len - 1].to_vec();
                    //         swap_instructions[i].instruction.accounts = new_vec
                    //     }
                    //     DexLabel::ORCA => {
                    //         //Dex Not implemented
                    //     }
                    //     DexLabel::RAYDIUM_CLMM => {
                    //         //Dex Not implemented
                    //     }
                    // }
                }
                false => {
                    error!("❌ No LUT address already crafted for the market {:?}, the tx can revert...", si.market.as_ref().unwrap().address);
                }
            }
        } else {
            info!("Skip get LUT table for non swap instruction: {:?}", si.details);
            continue
        }
    }
    
    let si_details: Vec<String> = swap_instructions.clone().into_iter().map(|instruc_details| instruc_details.details).collect();
    info!("📋 Swap instructions Details: {:#?}", si_details);
    info!("Swap instructions : {:?}", swap_instructions);

    //Get previously crafted LUT address
    let mut vec_address_lut: Vec<AddressLookupTableAccount> = Vec::new();

    for lut_address in lut_addresses {
        let raw_lut_account = rpc_client.get_account(&lut_address)?;
        let address_lookup_table = AddressLookupTable::deserialize(&raw_lut_account.data)?;
        let address_lookup_table_account = AddressLookupTableAccount {
            key: lut_address,
            addresses: address_lookup_table.addresses.to_vec(),
        };
        // println!("Address in lookup_table: {:#?}", address_lookup_table_account);
        println!("Address in lookup_table: {}", address_lookup_table_account.addresses.len());
        vec_address_lut.push(address_lookup_table_account);
    }

    // println!("instructions: {:?}", instructions);
    let mut instructions: Vec<Instruction> = swap_instructions.clone().into_iter().map(|instruc_details| instruc_details.instruction).collect();

    let commitment_config = CommitmentConfig::confirmed();
    let tx = VersionedTransaction::try_new(
        VersionedMessage::V0(v0::Message::try_compile(
            &payer.pubkey(),
            &instructions,
            &vec_address_lut,
            rpc_client.get_latest_blockhash_with_commitment(commitment_config).expect("❌ Error in get latest blockhash").0,
        )?),
        &[&payer],
    )?;

    //Simulate
    let config = RpcSimulateTransactionConfig {
        sig_verify: true,
        commitment: Some(commitment_config),
        .. RpcSimulateTransactionConfig::default()
    };
    
    //For loop simulation
    // for i in 0..9 {
    //     let result = rpc_client.simulate_transaction_with_config(&tx, config.clone()).unwrap().value;
    //     if result.clone().logs.unwrap().len() == 0 {
    //         error!("❌ Get out! Simulate Error: {:#?}", result.err);
    //         return Ok(())
    //     } else {
    //         info!("🧾 Simulate Tx Ata/Extend Logs: {:#?}", result.logs);
    //     }
    // }

    let result = rpc_client.simulate_transaction_with_config(&tx, config).unwrap().value;
    let logs_simulation = result.clone().logs.unwrap();
    let last_logs_simulation = &logs_simulation[logs_simulation.len() - 1];
    println!("last_logs_simulation: {}", last_logs_simulation);
    if logs_simulation.len() == 0 {
        error!("❌ Get out! Simulate Error: {:#?}", result.err);
        return Ok(())
    } else {
        info!("🧾 Simulate Tx Ata/Extend Logs: {:#?}", result.logs);
    }

    let result_cu: u64 = result.units_consumed.unwrap();
    let result_cu: u64 = 150000;
    info!("🔢 Computed Units: {}", result_cu);

    let fees = rpc_client.get_recent_prioritization_fees(&vec![])?;
    let average_fees = average(fees.iter().map(|iter| iter.prioritization_fee).collect());
    info!("🔢 Average Prioritization fees price: {}", average_fees);

    let compute_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(result_cu as u32);
    let priority_fees_ix = ComputeBudgetInstruction::set_compute_unit_price(100);
    instructions[0] = priority_fees_ix;
    instructions[1] = compute_budget_ix;


    //Send transaction
    if simulate_or_send == SendOrSimulate::Send {
        let transaction_config: RpcSendTransactionConfig = RpcSendTransactionConfig {
            skip_preflight: true,
            //Confirmed give more accurate result: https://www.helius.dev/blog/how-to-land-transactions-on-solana#blockhash
            preflight_commitment: Some(CommitmentLevel::Confirmed),
            encoding: Some(UiTransactionEncoding::Base58),
            max_retries: Some(0),
            min_context_slot: None,

        };
 
        let new_payer: Keypair = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");
        let txn: Transaction = Transaction::new_signed_with_payer(
            &instructions,
            Some(&new_payer.pubkey()),
            &vec![&new_payer],
            rpc_client.get_latest_blockhash_with_commitment(commitment_config).expect("❌ Error in get latest blockhash").0,
        );
        println!("Rpc http address: {}", rpc_client.url());
        // let signature = rpc_client.send_transaction_with_config(
        //     &txn,
        //     transaction_config
        // ).unwrap();
        
        let non_blocking_rpc_client = solana_client::nonblocking::rpc_client::RpcClient::new(env.rpc_url_tx.clone());
        let arc_rpc_client = Arc::new(non_blocking_rpc_client);
        let connection_cache = ConnectionCache::new_quic("connection_cache_cli_program_quic", 1);
        let signer: [Arc<dyn Signer>; 1] = [Arc::new(new_payer) as Arc<dyn Signer>];

        let iteration_number = 2;
        let mut iteration_counter = 0;
        let transaction_errors = if let ConnectionCache::Quic(cache) = connection_cache {
            let tpu_client = solana_client::nonblocking::tpu_client::TpuClient::new_with_connection_cache(
                arc_rpc_client.clone(),
                &env.wss_rpc_url,
                TpuClientConfig::default(),
                cache,
            )
            .await?;
            let error_tx = send_and_confirm_transactions_in_parallel(
                arc_rpc_client,
                Some(tpu_client),
                &[txn.message],
                &signer,
                SendAndConfirmConfig {
                    resign_txs_count: Some(iteration_number),
                    with_spinner: true,
                },
            )
            .await
            .map_err(|err| format!("Data writes to account failed: {err}")).unwrap_or_default()
            .into_iter()
            .map(|err| format!("Data writes to account failed: {:?}", err))
            // .flatten()
            .collect::<String>();
            info!("❌ Swap transaction is not executed: {:?}", error_tx);
            iteration_counter += 1;
        };
        // if iteration_counter >= iteration_number {
        //     error!("❌ Swap transactions sended {} times, and all fails", iteration_counter);
        // } else {
        //     info!("✅ Swap transaction is well executed!");
        // }
    }
    Ok(())
}

pub async fn create_ata_extendlut_transaction(chain: ChainType, simulate_or_send: SendOrSimulate, transaction_infos: SwapPathResult, lut_address: Pubkey, tokens: Vec<Pubkey>) -> Result<()> {
    info!("🔄 Create ATA/Extend LUT transaction.... ");
    
    let env = Env::new();
    let rpc_url = if chain.clone() == ChainType::Mainnet { &env.rpc_url_tx } else { &env.devnet_rpc_url };
    let rpc_client: RpcClient = RpcClient::new(rpc_url);

    let payer: Keypair = read_keypair_file(env.payer_keypair_path.clone()).expect("Wallet keypair file not found");
    info!("💳 Wallet {:#?}", payer.pubkey());

    let mut vec_pda_instructions: Vec<Instruction> = Vec::new();

    //Create Pda/Ata accounts
    for token in tokens {
        let pda_user_token = get_associated_token_address(
            &payer.pubkey(),
            &token,
        );
        match rpc_client.get_account(&pda_user_token) {
            Ok(account) => {
                info!("🟢 PDA for {} already exist !", token);
            }
            Err(error) => {
                info!("👷‍♂️ PDA creation...");
                let create_pda_instruction = create_associated_token_account(
                    &payer.pubkey(),
                    &payer.pubkey(),
                    &token,
                    &spl_token::id()
                );
                vec_pda_instructions.push(create_pda_instruction);
            }
        }
    }

    // Create the extend LUT instructions
    let mut swap_instructions: Vec<InstructionDetails> = construct_transaction(transaction_infos).await;
    // println!("swap_instructions len: {:?}", swap_instructions.len());
    // println!("swap_instructions: {:?}", swap_instructions);
    //Check if lookup table is already exist
    for (i, instruction) in swap_instructions.clone().iter().enumerate() {
        let market_addr = instruction.market.as_ref().unwrap().address;
        let (lut_exist, lut_address) = get_lut_address_for_market(market_addr, false).unwrap();
        if lut_exist == true {
            info!("🟢 Lookup already exist for {} !", market_addr);
            // println!("i: {}", i);
            // println!("swap_instructions[{}]: {:?}", i, swap_instructions[i]);
            swap_instructions.remove(0);
            continue;
        } else {  
            info!("👷‍♂️ Extend lookup added for: {}", market_addr);
            // println!("i: {}", i);
            // println!("swap_instructions[{}]: {:?}", i, swap_instructions[i]);
        }
    }
    if swap_instructions.len() == 0 && vec_pda_instructions.len() == 0 {
        info!("➡️ No ATA/Extend lookup transaction needed");
        return Ok(())   
    }
    let mut vec_details_extend_instructions: Vec<InstructionDetails> = Vec::new();
    // println!("AFTER");
    // println!("swap_instructions: {:?}", swap_instructions);
    
    for instr in swap_instructions.clone() {
        if instr.instruction.accounts.len() == 0 {
            continue;
        }
        if instr.clone().market.is_none() {
            error!("❌ Instruction market is None");
            return Ok(());
        }
        let accounts: Vec<Pubkey> = instr.instruction.accounts.iter().map(|account| account.pubkey).collect();
        // println!("Accounts: {:?}", accounts);

        let extend_instruction = extend_lookup_table(
            lut_address, 
            payer.pubkey(), 
            Some(payer.pubkey()), 
            accounts
        );
        vec_details_extend_instructions.push(InstructionDetails{instruction: extend_instruction, details: "Extend Instruction".to_string(), market: instr.clone().market});
        
        println!("Extend LUT instruction for market: {:?}", instr.clone().market);
        //Break because 2 extends lookup instruction not fit in 1 tx
        break;
    }

    let compute_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(1_400_000);
    let priority_fees_ix = ComputeBudgetInstruction::set_compute_unit_price(0);
    let compute_budget_instruction = vec![priority_fees_ix, compute_budget_ix];

    let vec_extend_instructions: Vec<Instruction> = vec_details_extend_instructions.clone().into_iter().map(|instruc_details| instruc_details.instruction).collect();

    let mut vec_all_instructions: Vec<Instruction> = vec![compute_budget_instruction.clone(), vec_pda_instructions.clone(), vec_extend_instructions].concat();

    let commitment_config = CommitmentConfig::confirmed();
    let txn_simulate: Transaction = Transaction::new_signed_with_payer(
        &vec_all_instructions,
        Some(&payer.pubkey()),
        &vec![&payer],
        rpc_client.get_latest_blockhash_with_commitment(commitment_config).expect("❌ Error in get latest blockhash").0,
    );
    // println!("Tx size: {:?}", txn);

    //Simulate
    let config = RpcSimulateTransactionConfig {
        sig_verify: true,
        commitment: Some(commitment_config),
        .. RpcSimulateTransactionConfig::default()
    };
    
    let result = rpc_client.simulate_transaction_with_config(&txn_simulate, config).unwrap().value;
    if result.clone().logs.unwrap().len() == 0 {
        error!("❌ Get out! Simulate Error: {:#?}", result.err);
        return Ok(())
    } else {
        info!("🧾 Simulate Tx Ata/Extend Logs: {:#?}", result.logs);
    }

    let result_cu: u64 = result.units_consumed.unwrap();
    info!("🔢 Computed Units: {}", result_cu);

    let fees = rpc_client.get_recent_prioritization_fees(&vec![])?;
    let average_fees = average(fees.iter().map(|iter| iter.prioritization_fee).collect());
    info!("🔢 Average Prioritization fees price: {}", average_fees);

    let compute_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(result_cu as u32);
    let priority_fees_ix = ComputeBudgetInstruction::set_compute_unit_price(10);
    vec_all_instructions[0] = priority_fees_ix;
    vec_all_instructions[1] = compute_budget_ix;

    //Send transaction
    if simulate_or_send == SendOrSimulate::Send {
        let transaction_config: RpcSendTransactionConfig = RpcSendTransactionConfig {
            skip_preflight: false,
            //Confirmed give more accurate result: https://www.helius.dev/blog/how-to-land-transactions-on-solana#blockhash
            preflight_commitment: Some(CommitmentLevel::Confirmed),
            encoding: Some(UiTransactionEncoding::Base58),
            max_retries: Some(0),
            min_context_slot: None,

        };
 
        let new_payer: Keypair = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");
        let txn: Transaction = Transaction::new_signed_with_payer(
            &vec_all_instructions,
            Some(&new_payer.pubkey()),
            &vec![&new_payer],
            rpc_client.get_latest_blockhash_with_commitment(commitment_config).expect("❌ Error in get latest blockhash").0,
        );
        println!("Rpc http address: {}", rpc_client.url());
        // let signature = rpc_client.send_transaction_with_config(
        //     &txn,
        //     transaction_config
        // ).unwrap();
        
        let non_blocking_rpc_client = solana_client::nonblocking::rpc_client::RpcClient::new(env.rpc_url_tx.clone());
        let arc_rpc_client = Arc::new(non_blocking_rpc_client);
        let connection_cache = ConnectionCache::new_quic("connection_cache_cli_program_quic", 1);
        let signer: [Arc<dyn Signer>; 1] = [Arc::new(new_payer) as Arc<dyn Signer>];

        let iteration_number = 5;
        let mut iteration_counter = 0;
        let transaction_errors = if let ConnectionCache::Quic(cache) = connection_cache {
            let tpu_client = solana_client::nonblocking::tpu_client::TpuClient::new_with_connection_cache(
                arc_rpc_client.clone(),
                &env.wss_rpc_url,
                TpuClientConfig::default(),
                cache,
            )
            .await?;
            let error_tx = send_and_confirm_transactions_in_parallel(
                arc_rpc_client,
                Some(tpu_client),
                &[txn.message],
                &signer,
                SendAndConfirmConfig {
                    resign_txs_count: Some(iteration_number),
                    with_spinner: true,
                },
            )
            .await
            .map_err(|err| format!("Data writes to account failed: {err}")).unwrap()
            .into_iter()
            .map(|err| format!("Data writes to account failed: {:?}", err))
            // .flatten()
            .collect::<String>();
            println!("❌ Ata/Extend transaction is not executed: {:?}", error_tx);
            iteration_counter += 1;
        };
        if iteration_counter >= iteration_number {
            error!("❌ Ata/Extend transaction sended {} times, and all fails", iteration_counter);
        } else {
            for details_instruction in vec_details_extend_instructions {
                let _ = write_lut_for_market(details_instruction.market.unwrap().address, lut_address, false);
            }
            info!("✅ Ata/Extend transaction is well executed");
        }

        // if chain.clone() == ChainType::Devnet {
        //     info!("https://explorer.solana.com/tx/{}?cluster=devnet", signature);
        // } else {
        //     info!("https://explorer.solana.com/tx/{}", signature);
        // }
        // let tx_executed = check_tx_status(commitment_config, chain.clone(), signature).await?;
        // if tx_executed {
        //     for details_instruction in vec_details_extend_instructions {
        //         let _ = write_lut_for_market(details_instruction.market.unwrap().address, lut_address, false);
        //     }
        //     info!("✅ Ata/Extend transaction is well executed");
        //     break;
        // } else {
        //     info!("❌ Ata/Extend transaction is not executed");
        // }

    }
    
    Ok(())
}
pub async fn construct_transaction(transaction_infos: SwapPathResult) -> Vec<InstructionDetails> {
    
    let mut swap_instructions: Vec<InstructionDetails> = Vec::new();
    
    for (i, route_sim) in transaction_infos.route_simulations.clone().iter().enumerate() {
        match route_sim.dex_label {
            DexLabel::METEORA => {
                let swap_params: SwapParametersMeteora = SwapParametersMeteora{
                    lb_pair: from_str(transaction_infos.route_simulations[i].pool_address.as_str()).unwrap(),
                    amount_in: transaction_infos.route_simulations[i].amount_in,
                    swap_for_y: transaction_infos.route_simulations[i].token_0to1,
                    input_token: from_str(transaction_infos.route_simulations[i].token_in.as_str()).unwrap(),
                    output_token: from_str(transaction_infos.route_simulations[i].token_out.as_str()).unwrap(),
                    minimum_amount_out: transaction_infos.route_simulations[i].estimated_amount_out.parse().unwrap()
                };
                let result = construct_meteora_instructions(swap_params.clone()).await;
                if result.len() == 0 {
                    let empty_array: Vec<InstructionDetails> = Vec::new();
                    error!("Error in Meteora Instruction");
                    return empty_array;
                }
                for instruction in result {
                    swap_instructions.push(instruction);
                }
            }
            DexLabel::RAYDIUM => {
                let swap_params: SwapParametersRaydium = SwapParametersRaydium{
                    pool: from_str(transaction_infos.route_simulations[i].pool_address.as_str()).unwrap(),
                    input_token_mint: from_str(route_sim.token_in.as_str()).unwrap(),
                    output_token_mint: from_str(route_sim.token_out.as_str()).unwrap(),
                    amount_in: transaction_infos.route_simulations[i].amount_in,
                    swap_for_y: transaction_infos.route_simulations[i].token_0to1,
                    min_amount_out: transaction_infos.route_simulations[i].estimated_amount_out.parse().unwrap()
                };
                let result = construct_raydium_instructions(swap_params);
                if result.len() == 0 {
                    let empty_array: Vec<InstructionDetails> = Vec::new();
                    error!("Error in Raydium Instruction");
                    return empty_array;
                }
                for instruction in result {
                    swap_instructions.push(instruction);
                }
            }
            DexLabel::RAYDIUM_CLMM => {
                info!("⚠️ RAYDIUM_CLMM TX NOT IMPLEMENTED");
            }
            DexLabel::ORCA_WHIRLPOOLS => {
                let swap_params: SwapParametersOrcaWhirpools = SwapParametersOrcaWhirpools{
                    whirpools: from_str(transaction_infos.route_simulations[i].pool_address.as_str()).unwrap(),
                    input_token: from_str(route_sim.token_in.as_str()).unwrap(),
                    output_token: from_str(route_sim.token_out.as_str()).unwrap(),
                    amount_in: transaction_infos.route_simulations[i].amount_in,
                    minimum_amount_out: transaction_infos.route_simulations[i].estimated_amount_out.parse().unwrap()
                };
                let result = construct_orca_whirpools_instructions(swap_params).await;
                // Return len 0 to handle error case in swap
                if result.len() == 0 {
                    let empty_array: Vec<InstructionDetails> = Vec::new();
                    error!("Error in Orca_Whirpools Instruction");
                    return empty_array;
                }
                for instruction in result {
                    swap_instructions.push(instruction);
                }
            }
            DexLabel::ORCA => {
                info!("⚠️ ORCA TX NOT IMPLEMENTED");
            }
        }
    }
    return swap_instructions;
}

pub async fn create_lut(chain: ChainType) -> Result<()> {
    info!("🆔 Create/Send LUT transaction....");
    let env = Env::new();
    let rpc_url = if chain == ChainType::Mainnet { env.rpc_url } else { env.devnet_rpc_url };
    let rpc_client: RpcClient = RpcClient::new(rpc_url);
    let payer: Keypair = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");

    //Create Address Lookup Table (LUT acronym)
    let slot = rpc_client.get_slot_with_commitment(CommitmentConfig::finalized()).expect("Error in get slot");
    let (create_lut_instruction, lut_address) = create_lookup_table(
        payer.pubkey(),
        payer.pubkey(),
        slot - 200
    );

    let txn_lut = Transaction::new_signed_with_payer(
        &vec![create_lut_instruction.clone()],
        Some(&payer.pubkey()),
        &vec![&payer],
        rpc_client.get_latest_blockhash().expect("Error in get latest blockhash"),
    );
    let transaction_config: RpcSendTransactionConfig = RpcSendTransactionConfig {
        skip_preflight: false,
        .. RpcSendTransactionConfig::default()
    };

    let signature = rpc_client.send_transaction_with_config(
        &txn_lut,
        transaction_config
    ).unwrap();

    if chain == ChainType::Devnet {
        info!("https://explorer.solana.com/tx/{}?cluster=devnet", signature);
    } else {
        info!("https://explorer.solana.com/tx/{}", signature);
    }
    let commitment_config = CommitmentConfig::confirmed();
    let tx_confirmed = check_tx_status(commitment_config, chain, signature).await?;
    if tx_confirmed {
        info!("✅ Address LUT is well created");
        info!("🧾 Address LUT {:#?}", lut_address);
    } else {
        info!("❌ Address LUT is not created");
        info!("🧾 Address LUT {:#?}", lut_address);
    }

    Ok(())
}

pub async fn is_available_lut(chain: ChainType, lut_address: Pubkey) -> Result<bool> {
    info!("🚚 Check if LUT address is available to craft a transaction...");
    let env = Env::new();
    let rpc_url = if chain == ChainType::Mainnet { env.rpc_url } else { env.devnet_rpc_url };
    let rpc_client: RpcClient = RpcClient::new(rpc_url);
    let payer: Keypair = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");

    let raw_lut_account = rpc_client.get_account(&lut_address)?;
    let address_lookup_table = AddressLookupTable::deserialize(&raw_lut_account.data)?;
    
    let lut_length = address_lookup_table.addresses.len();
    if lut_length < 210 {
        return Ok(true)
    } else {
        return Ok(false)
    }
}

pub fn get_lut_address_for_market(market: Pubkey, is_test: bool) -> Result<(bool, Option<Pubkey>)> {
    let mut path = "";
    if is_test {
        path = "src/transactions/cache/lut_addresses_test.json";
    } else {
        path = "src/transactions/cache/lut_addresses.json";
    }
    let file_read = OpenOptions::new().read(true).open(path)?;
    let mut lut_file: VecLUTFile = serde_json::from_reader(&file_read).unwrap();
    let lut_address = lut_file.value.iter().find(|iteration| &from_str(iteration.market.as_str()).unwrap() == &market);
    match lut_address  {
        Some(value) => {
            return Ok((true, Some(from_str(value.lut_address.as_str()).unwrap())))
        }
        None => {
            return Ok((false, None))
        }
    }
}

pub fn write_lut_for_market(market:Pubkey, lut_address: Pubkey, is_test: bool) -> Result<()> {
    let mut path = "";
    if is_test {
        path = "src/transactions/cache/lut_addresses_test.json";
    } else {
        path = "src/transactions/cache/lut_addresses.json";
    }
    let file_exist = Path::new(path).is_file();
    if !file_exist {
        File::create(path)?;

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut writer = BufWriter::new(&file);

        let mut lut_file = VecLUTFile{value: Vec::new()};
        let field = LUTFile{market: market.to_string(), lut_address: lut_address.to_string()};
        lut_file.value.push(field);
        println!("LUT File: {:?}", lut_file);
        writer.write_all(serde_json::to_string(&lut_file)?.as_bytes())?;
        writer.flush()?;
        info!("Data written to 'lut_addresses.json' successfully.");
    } else {
        let file_read = OpenOptions::new().read(true).write(true).open(path)?;
        let mut lut_file: VecLUTFile = serde_json::from_reader(&file_read).unwrap();
        
        let file_write = OpenOptions::new().read(true).write(true).truncate(true).open(path)?;
        let mut writer = BufWriter::new(&file_write);
        // let mut lut_file = VecLUTFile{value: Vec::new()};
        let field = LUTFile{market: market.to_string(), lut_address: lut_address.to_string()};
        lut_file.value.push(field);
        // println!("LUT File: {:?}", lut_file);
        writer.write_all(serde_json::to_string(&lut_file)?.as_bytes())?;
        writer.flush()?;
        info!("Data written to 'lut_addresses.json' successfully.");
    }

    Ok(())
}

////////////////////

#[derive(Debug, Clone)]
pub struct InstructionDetails {
    pub instruction: Instruction,
    pub details: String,
    pub market: Option<MarketInfos>
}
#[derive(Debug, Clone)]
pub struct MarketInfos {
    pub dex_label: DexLabel,
    pub address: Pubkey,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecLUTFile {
    pub value: Vec<LUTFile>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LUTFile {
    pub market: String,
    pub lut_address: String,
}

pub enum TransactionType {
    CreateLUT,
    CreateSwap,
}
#[derive(PartialEq)]
pub enum SendOrSimulate {
    Simulate,
    Send,
}

#[derive(PartialEq, Clone)]
pub enum ChainType {
    Mainnet,
    Devnet,
}
