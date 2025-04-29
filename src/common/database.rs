use log::info;
use mongodb::bson::{doc, Document};
use mongodb::Collection;
use mongodb::{Client as MongoDbCLient, options::ClientOptions};
use anyhow::Result;

use crate::arbitrage::types::{SwapPathResult, VecSwapPathSelected};

pub async fn insert_swap_path_result_collection(collection_name: &str, sp_result: SwapPathResult) -> Result<()> {
    let db_name = "MEV_Bot";
    let client_options = ClientOptions::parse("mongodb://localhost:27017").await.unwrap();
    let client = MongoDbCLient::with_options(client_options).unwrap();

    let db = client.database(db_name);
    let coll: Collection<SwapPathResult> = db.collection::<SwapPathResult>(collection_name);
    

    coll.insert_one(sp_result, None).await.unwrap();
    info!("📊 {} writed in DB", collection_name);
    Ok(())
}
pub async fn insert_vec_swap_path_selected_collection(collection_name: &str, best_paths_for_strat: VecSwapPathSelected) -> Result<()> {
    for bp in best_paths_for_strat.value.iter().enumerate() {

    }
    let db_name = "MEV_Bot";

    let client_options = ClientOptions::parse("mongodb://localhost:27017").await.unwrap();
    let client = MongoDbCLient::with_options(client_options).unwrap();

    let db = client.database(db_name);
    let coll: Collection<VecSwapPathSelected> = db.collection::<VecSwapPathSelected>(collection_name);
    
    coll.insert_one(best_paths_for_strat, None).await.unwrap();
    info!("📊 {} writed in DB", collection_name);

    Ok(())
}