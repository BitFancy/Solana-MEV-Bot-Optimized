import {
    getLatestCoin,
    token_sell,
    getTokenPrice,
    getSPLTokens,
    token_buy,
    SPL_token_balance,
    getPnl,
    getsplTokeninfo_bird,
    wallet_positions_bird,
    getTokenPrice_sol,
    gettokeninfo,
  } from "./fuc.js";
  
  import dotenv from "dotenv";
  import { PublicKey, Keypair } from "@solana/web3.js";
  import { readFile, writeFile } from "fs/promises";
  import fs from "fs"; // Add fs import for readFileSync
  import bs58 from "bs58";
  import { sleep } from "./fuc.js";
  import chalk from "chalk";
  import { bot } from "./bot.js";
  import { timeStamp } from "console";
  import { getPriceForMint, isPriceStreamRunning, priceStreamPromise, stopPriceStream } from "./price_stream.js";
  import { threadId } from "worker_threads";
  
  dotenv.config();
  const BUY_AMOUNT = process.env.BUY_AMOUNT;
  
  // Initialize keypair from file
  let keypair;
  try {
    const keyData = await readFile("bot.json", "utf8");
    if (!keyData) {
      throw new Error("Empty keypair file");
    }
    const secretKey = new Uint8Array(JSON.parse(keyData));
    keypair = Keypair.fromSecretKey(secretKey);
  } catch (error) {
    console.error("Error reading keypair file:", error);
    process.exit(1);
  }
  
  // Set up key constants
  export const PUBLIC_KEY = keypair.publicKey.toString();
  
  // Add a global flag to control auto sell monitoring
  export let isAutoSellRunning = false;
  
  // Function to stop auto sell monitoring
  export const stopAutoSell = () => {
    isAutoSellRunning = false;
    console.log(chalk.red("Auto sell monitoring stopped"));
  };
  const tokenSellFlags = new Map();
  const tokenbuyFlags = new Map();
  // await getBalance();
  export const monitoring_sell_all = async () => {
    try {
      const spl_tokens = await wallet_positions_bird(PUBLIC_KEY);
      console.log(chalk.yellow("spl_tokens", spl_tokens));
  
      // Run all sell operations concurrently
      await Promise.all(spl_tokens.map((token) => token_sell(token.address, 100, token.uiAmount, token.decimals)));
  
      console.log("sell running...");
  
      await sleep(2000); // Check every 2 seconds
    } catch (error) {
      console.error("Error monitoring positions:", error);
      await sleep(5000); // Wait before retrying on error
    }
  };
  export const monitoring_buy_one = async (token, chat_id) => {
    try {
      // Check blacklist
      const blacklistData = await readFile("blacklist.json", "utf8");
      const blacklistedTokens = blacklistData.trim() ? JSON.parse(blacklistData) : [];
  
      if (blacklistedTokens.includes(token.address)) {
        return;
      }
    } catch (err) {
      console.warn("❌ Could not read blacklist:", err);
      if (err.code === "ENOENT") {
        console.log("No blacklist.json file found, continuing...");
      }
    }
  
    // Initialize token flags if not exists
    if (!tokenbuyFlags.has(token.address)) {
      tokenbuyFlags.set(token.address, {
        completedIntervals: new Set(),
        lastSellTime: 0,
      });
    }
    const tokenFlags = tokenbuyFlags.get(token.address);
  
    try {
      const buyTime = new Date(token.lastUpdate).getTime();
      const timeElapsed = (Date.now() - buyTime) / 1000;
  
      console.log("Token address:", token.address);
      console.log("Buy time:", buyTime);
      console.log("Time elapsed:", timeElapsed);
      console.log("Buy count:", token.buyCount);
  
      // Only attempt to buy if:
      // 1. Less than 2 buys have occurred
      // 2. Less than 15 seconds have elapsed since last update
      // 3. Current price is 20% below bought price
      if (token.buyCount < 3 && timeElapsed < 10) {
        const currentPrice = await getPriceForMint(token.address);
        const priceThreshold = token.boughtPrice * 0.8;
        
        console.log("🟢🟢🟢Current price:", currentPrice);
        console.log("🟢🟢🟢Price threshold:", priceThreshold);
        console.log("🟢🟢🟢Completed intervals:", tokenFlags.completedIntervals);
        
        if (currentPrice === null) {
          console.log("Current price is null, skipping...");
          return;
        }
        if (currentPrice < priceThreshold && !tokenFlags.completedIntervals.has(0)) {
          console.log(`✅🟢🟢  Price below threshold - Current: ${currentPrice}, Threshold: ${priceThreshold}`);
  con
  nst
  const
  txid = await token_buy(token.address, BUY_AMOUNT);
          if (txid) {
            tokenFlags.completedIntervals.add(0);
            tokenFlags.lastSellTime = Date.now();
            console.log(`🔵 Token bought ${BUY_AMOUNT} SOL - Transaction: ${txid}\n Address: ${token.address}`);
  
            try {
              const message = ✅ Bought: ${token.address}\n💰 Amount: ${BUY_AMOUNT} SOL\n🔗 <a href="https://solscan.io/tx/${txid}">View Transaction on Explorer</a>\n💼 <a href="https://gmgn.ai/sol/token/${token.address}">GMGN Wallet</a>;
              await bot.sendMessage(chat_id, message, { parse_mode: "HTML" });
            } catch (telegramError) {
              console.error("Failed to send Telegram notification:", telegramError);
            }
            return txid;
          }
        }
      }
      return null;
    } catch (error) {
      console.error("Error in monitoring buy:", error);
      await sleep(2000);
      return null;
    }
  };
  
  // Initialize sell flags outside the function for persistence across calls
  
  
  const sellToken = async (token, chat_id) => {
    try {
      // Check blacklist
      const blacklistData = await readFile("blacklist.json", "utf8");
      const blacklistedTokens = blacklistData.trim() ? JSON.parse(blacklistData) : [];
  
      if (blacklistedTokens.includes(token.address)) {
        // console.log(`⚠️ Token ${token.symbol} is blacklisted, skipping sell`);
        return;
      }
    } catch (err) {
      console.warn("❌ Could not read blacklist:", err);
      if (err.code === "ENOENT") {
        console.log("No blacklist.json file found, continuing...");
      }
    }
    // if (token.amount < 1) {
    //   return;
    // }
    let currentPrice;
    let pnl;
    let timeElapsed;
    try {
      currentPrice = await getPriceForMint(token.address);
      //  console.log("Got current price:", currentPrice);
      pnl = ((currentPrice - token.boughtPrice) * 100) / token.boughtPrice;
    } catch (err) {
      console.error("Error calculating PNL:", err);
      return;
    }
    // console.log("token details", token);
    
    try {
       const buyTime = new Date(token.lastUpdate).getTime(); 
       timeElapsed = (Date.now() - buyTime) / 1000;
      } catch (err) {
        console.error("Error calculating time elapsed:", err);
        return;
      }
      
      const tokenFlags = tokenSellFlags.get(token.address);
      // Helper function for selling and notifications
      const sellAndNotify = async (percentage_amount, timeDescription) => {
        try {
        console.log(`sellAndNotify_____${percentage_amount}___${token.amount}___________________________`);
        let txid = await token_sell(token.address, percentage_amount, token.amount/(10**token.decimals), token.decimals);
        console.log("Got transaction ID:", txid);
        
        const pnl_amount_sol = (percentage_amount*(token.amount * currentPrice * pnl) / 10000).toFixed(4);
        console.log("Calculated PNL amount:", pnl_amount_sol);
        
        if (percentage_amount === 100){
          tokenFlags.topPnl = 0;
          tokenFlags.lastSellTime = 0;
          tokenFlags.completedIntervals = new Set();
        }
        if (txid) {
          try {
            const message = `🟥Sell:\n\`${token.address}\`\n💰 ${pnl < 0 ? "🔴" : "🟢"} PNL: ${pnl.toFixed(
              2
            )}% PNL Amount: ${pnl_amount_sol} SOL\n📊 Selling ${(percentage_amount*token.amount/100).toFixed(
              2
            )} tokens (${percentage_amount}% ${timeDescription})\n🔗 <a href="https://solscan.io/tx/${txid}">View Transaction</a>`;
            await bot.sendMessage(chat_id, message, { parse_mode: "HTML" });
          } catch (telegramError) {
            console.error("Failed to send Telegram notification:", telegramError);
          }
        }
      } catch (err) {
        console.error("Error in sellAndNotify:", err);
      }
    };
    
    try {
      // Initialize token flags if not exists
      if (!tokenSellFlags.has(token.address)) {
        tokenSellFlags.set(token.address, {
          lastSellTime: 0,
          completedIntervals: new Set(),
        });
      }
      if (!tokenFlags.topPnl || pnl > tokenFlags.topPnl) {
        tokenFlags.topPnl = pnl;
        console.log(chalk.bgCyanBright(`New top PNL for
  ${token.address}: ${pnl}%`));
        
      }
      // const during_time = 8;
      // const sell_start_time = 180;
      console.log(chalk.greenBright(`T->${timeElapsed}-pnl:${pnl.toFixed(3)}<---${token.address.substring(0, 8)}...-top_pnl-->${tokenFlags.topPnl.toFixed(4)}`));
      console.log(tokenFlags.completedIntervals.has("stop_loss_15"))
      // Stop loss and take profit conditions
      // if (pnl < 0 && !tokenFlags.completedIntervals.has("stop_loss_0")  && timeElapsed > 6) {
      //   console.log("Executing stop loss at 0%");
      //   await sellAndNotify(100, "Stop loss 0%");
      //   tokenFlags.completedIntervals.add("stop_loss_0");
      //   tokenFlags.lastSellTime = Date.now();
      //   return;
      // }
      if (pnl < 0 && !tokenFlags.completedIntervals.has("stop_loss_15") ) {
        console.log("Executing stop loss at -15%");
        await sellAndNotify(100, "Stop loss -15%");
        tokenFlags.completedIntervals.add("stop_loss_15");
        tokenFlags.lastSellTime = Date.now();
        return;
      }
      // if (pnl < -10 && !tokenFlags.completedIntervals.has("stop_loss_10") ) {
      //   console.log("Executing stop loss at -10%");
      //   await sellAndNotify(100, "Stop loss -10%");
      //   tokenFlags.completedIntervals.add("stop_loss_10");
      //   tokenFlags.lastSellTime = Date.now();
      //   return;
      // }
      if (pnl < 0 && !tokenFlags.completedIntervals.has("stop_loss_15") && timeElapsed > 8 || pnl < 5 && !tokenFlags.completedIntervals.has("stop_loss_15") && timeElapsed > 15) {
        console.log("Executing time-based exit at 15 s");
        await sellAndNotify(100, "Time-based exit after 15 s");
        tokenFlags.completedIntervals.add("stop_loss_15");
        tokenFlags.lastSellTime = Date.now();
        return;
      }
    
      if (pnl < tokenFlags.topPnl*0.4 && tokenFlags.topPnl> 10) {
          console.log(`Executing trailing stop loss at ${tokenFlags.topPnl*0.3}% (70% drop from peak of ${tokenFlags.topPnl}%)`);
          await sellAndNotify(100, `Trailing stop loss: 70% drop from peak of ${tokenFlags.topPnl}%`);
          // tokenFlags.completedIntervals.add("take_profit_50");
          tokenFlags.lastSellTime = Date.now();
        return;
      }
    
  
      // Take profit conditions
      const takeProfitLevels = [
  
        { threshold: 2000, percentage: 100, flag: "tp_2000", message: "Take full profit at 2000%" },
        { threshold: 1500, percentage: 40, flag: "tp_1500", message: "Take full profit at 1500%" },
        { threshold: 1000, percentage: 40, flag: "tp_1000", message: "Take partial profit at 1000%" },
        { threshold: 800, percentage: 20, flag: "tp_800", message: "Take partial profit at 800%"},
        { threshold: 600, percentage: 20, flag: "tp_600", message: "Take partial profit at 600%",  },
        { threshold: 400, percentage: 20, flag: "tp_400", message: "Take full profit at 400%" },
        { threshold: 300, percentage: 20, flag: "tp_300", message: "Take partial profit at 300%" },
        { threshold: 250, percentage: 20, flag: "tp_250", message: "Take partial profit at 250%" },
        { threshold: 200, percentage: 20, flag: "tp_200", message: "Take partial profit at 200%" },
        { threshold: 120, percentage: 20, flag: "tp_120", message: "Take partial profit at 120%"},
        { threshold: 80, percentage: 20, flag: "tp_80", message: "Take partial profit at 80%",  },
        { threshold: 50, percentage: 10, flag: "tp_50", message: "Take partial profit at 50%",  },
        { threshold: 20, percentage:10, flag: "tp_20", message: "Take partial profit at 20%",  },
      ];
      
  
      for (const level of takeProfitLevels) {
        if (pnl > level.threshold && !tokenFlags.completedIntervals.has(level.flag)) {
          if (level.timeCheck !== undefined && !level.timeCheck) continue;
          console.log(`Executing take profit at ${level.threshold}%`);
          await sellAndNotify(level.percentage, level.message);
          tokenFlags.completedIntervals.add(level.flag);
          tokenFlags.lastSellTime = Date.now();
          if (level.percentage === 100){
  
            console.log(chalk.bgYellow(`Token amount is 0 for ${token.address}, resetting tracking
  values`));
  king values`));
  tokenFlags.completedIntervals = new Set();
            tokenFlags.topPnl = 0;
            tokenFlags.lastSellTime = 0;
            
            
            // Log the reset
            // fs.appendFileSync('pnl_log.txt', `${new Date().toISOString()} - Reset tracking for ${token.address}: Token amount is 0\n`);
            return; // Exit the function when token amount is 0
          }
        }
      }
      // Save top PNL value for the token
      
      // Check if token amount is 0, reset all values
      
      // Log the new top PNL as text
      // fs.appendFileSync('pnl_log.txt', `${new Date().toISOString()} - New top PNL for ${token.address}: ${pnl}%-top pnl-${tokenFlags.topPnl}\n`);
      
      // Retracement sell conditions - Sell when price drops by a percentage from the highest PNL
      let retracementLevels;
      if (timeElapsed > 30) {
        // Override retracementLevels based on current PNL
        if (pnl > 500) {
          retracementLevels = [
            { percentage: 3, threshold: 2000, flag: "retracement_2000_3", message: "Selling on 3% retracement from top (2000%+ PNL)", sellAmount: 100 },
            { percentage: 4, threshold: 1500, flag: "retracement_1500_4", message: "Selling on 4% retracement from top (1500%+ PNL)", sellAmount: 50 },
            { percentage: 5, threshold: 1000, flag: "retracement_1000_5", message: "Selling on 5% retracement from top (1000%+ PNL)", sellAmount: 40 },
            { percentage: 6, threshold: 800, flag: "retracement_800_6", message: "Selling on 6% retracement from top (800%+ PNL)", sellAmount: 35 },
            { percentage: 6, threshold: 700, flag: "retracement_700_6", message: "Selling on 6% retracement from top (700%+ PNL)", sellAmount: 35 },
            { percentage: 6, threshold: 600, flag: "retracement_600_6", message: "Selling on 6% retracement from top (600%+ PNL)", sellAmount: 30 },
            { percentage: 7, threshold: 500, flag: "retracement_500_7", message: "Selling on 7% retracement from top (500%+ PNL)", sellAmount: 30 },
            { percentage: 7, threshold: 400, flag: "retracement_400_7", message: "Selling on 7% retracement from top (400%+ PNL)", sellAmount: 30 },
            { percentage: 8, threshold: 300, flag: "retracement_300_8", message: "Selling on 8% retracement from top (300%+ PNL)", sellAmount: 20 },
            { percentage: 15, threshold: 20, flag: "retracement_500_15", message: "Selling on 15% retracement from top (500%+ PNL)", sellAmount: 100 }
          ];
        } else if (pnl > 200) {
          retracementLevels = [
            { percentage: 3, threshold: 2000, flag: "retracement_2000_3", message: "Selling on 3% retracement from top (2000%+ PNL)", sellAmount: 100 },
            { percentage: 4, threshold: 1500, flag: "retracement_1500_4", message: "Selling on 4% retracement from top (1500%+ PNL)", sellAmount: 50 },
            { percentage: 5, threshold: 1000, flag: "retracement_1000_5", message: "Selling on 5% retracement from top (1000%+ PNL)", sellAmount: 40 },
            { percentage: 6, threshold: 800, flag: "retracement_800_6", message: "Selling on 6% retracement from top (800%+ PNL)", sellAmount: 35 },
            { percentage: 6, threshold: 700, flag: "retracement_700_6", message: "Selling on 6% retracement from top (700%+ PNL)", sellAmount: 35 },
            { percentage: 8, threshold: 600, flag: "retracement_600_8", message: "Selling on 8% retracement from top (600%+ PNL)", sellAmount: 30 },
            { percentage: 10, threshold: 500, flag: "retracement_500_10", message: "Selling on 10% retracement from top (500%+ PNL)", sellAmount: 30 },
            { percentage: 10, threshold: 400, flag: "retracement_400_10", message: "Selling on 10% retracement from top (400%+ PNL)", sellAmount: 30 },
            { percentage: 10, threshold: 300, flag: "retracement_300_10", message: "Selling on 10% retracement from top (300%+ PNL)", sellAmount: 20 },
            { percentage: 10, threshold: 200, flag: "retracement_200_10", message: "Selling on 10% retracement from top (200%+ PNL)", sellAmount: 20 },
            { percentage: 12, threshold: 100, flag: "retracement_100_12", message: "Selling on 12% retrac
  ement from top (100%+ PNL)", sellAmount: 20 },
            { percentage: 20, threshold: 20, flag: "retracement_20_20", message: "Selling on 20% retracement from top (20%+ PNL)", sellAmount: 100 }
          ];
        } else {
          retracementLevels = [
            // High PNL thresholds with small retracement triggers
            { percentage: 3, threshold: 2000, flag: "retracement_2000_3", message: "Selling on 3% retracement from top (2000%+ PNL)", sellAmount: 100 },
            { percentage: 4, threshold: 1500, flag: "retracement_1500_4", message: "Selling on 4% retracement from top (1500%+ PNL)", sellAmount: 50 },
            { percentage: 5, threshold: 1000, flag: "retracement_1000_5", message: "Selling on 5% retracement from top (1000%+ PNL)", sellAmount: 40 },
            { percentage: 6, threshold: 800, flag: "retracement_800_6", message: "Selling on 6% retracement from top (800%+ PNL)", sellAmount: 35 },
            { percentage: 6, threshold: 700, flag: "retracement_700_6", message: "Selling on 6% retracement from top (700%+ PNL)", sellAmount: 35 },
            { percentage: 6, threshold: 600, flag: "retracement_600_6", message: "Selling on 6% retracement from top (600%+ PNL)", sellAmount: 30 },
            { percentage: 7, threshold: 500, flag: "retracement_500_7", message: "Selling on 7% retracement from top (500%+ PNL)", sellAmount: 30 },
            { percentage: 7, threshold: 400, flag: "retracement_400_7", message: "Selling on 7% retracement from top (400%+ PNL)", sellAmount: 30 },
            { percentage: 8, threshold: 300, flag: "retracement_300_8", message: "Selling on 8% retracement from top (300%+ PNL)", sellAmount: 20 },
            { percentage: 10, threshold: 200, flag: "retracement_200_10", message: "Selling on 10% retracement from top (200%+ PNL)", sellAmount: 15 },
            { percentage: 12, threshold: 100, flag: "retracement_100_12", message: "Selling on 12% retracement from top (100%+ PNL)", sellAmount: 15 },
            // Lower PNL thresholds with larger retracement triggers
            { percentage: 20, threshold: 50, flag: "retracement_50_20", message: "Selling on 20% retracement from top (50%+ PNL)", sellAmount: 10 },
            { percentage: 30, threshold: 30, flag: "retracement_30_30", message: "Selling on 30% retracement from top (30%+ PNL)", sellAmount: 10 },
            { percentage: 42, threshold: 20, flag: "retracement_20_42", message: "Selling on 42% retracement from top (20%+ PNL)", sellAmount: 100 }
          ];
        }
      }
      else {
        retracementLevels = [
          { percentage: 3, threshold: 2000, flag: "retracement_2000_3_pct", message: "Selling on 3% retracement from top (2000%+ PNL)", sellAmount: 100 },
          { percentage: 4, threshold: 1500, flag: "retracement_1500_4_pct", message: "Selling on 4% retracement from top (1500%+ PNL)", sellAmount: 50 },
          { percentage: 5, threshold: 1000, flag: "retracement_1000_5_pct", message: "Selling on 5% retracement from top (1000%+ PNL)", sellAmount: 40 },
          { percentage: 6, threshold: 800, flag: "retracement_800_6_pct", message: "Selling on 6% retracement from top (800%+ PNL)", sellAmount: 35 },
          { percentage: 6, threshold: 700, flag: "retracement_700_6_pct", message: "Selling on 6% retracement from top (700%+ PNL)", sellAmount: 35 },
          { percentage: 6, threshold: 600, flag: "retracement_600_6_pct", message: "Selling on 6% retracement from top (600%+ PNL)", sellAmount: 30 },
          { percentage: 7, threshold: 500, flag: "retracement_500_7_pct", message: "Selling on 7% retracement from top (500%+ PNL)", sellAmount: 30 },
          { percentage: 7, threshold: 400, flag: "retracement_400_7_pct", message: "Selling on 7% retracement from top (400%+ PNL)", sellAmount: 30 },
          { percentage: 8, threshold: 300, flag: "retracement_300_8_pct", message: "Selling on 8% retracement from top (300%+ PNL)", sellAmount: 20 },
          { percentage: 10, threshold: 200, flag: "retracement_200_10_pct", message: "Selling on 10% retracement from top (200%+ PNL)", sellAmount: 15 },
          { percentage: 12, threshold: 100, flag:
  "retracement_100_12_pct",
  rror);
        await sleep(1000); // Wait before retrying on error
      }
    }
  
    console.log(chalk.red("Auto sell monitoring loop exited"));
  };
  
  export const monitoring_autotrading = async (chat_id) => {
    // Set the flag to true when starting
    isAutoSellRunning = true;
    console.log(chalk.green("Auto trading started"));
    try {
      const sentMessage = await bot.sendMessage(chat_id, `🚀Auto trading started🚀`);
      setTimeout(async () => {
        try {
          await bot.deleteMessage(chat_id, sentMessage.message_id);
        } catch (error) {
          console.error("Error deleting message:", error);
        }
      }, 10000);
    } catch (telegramError) {
      console.error("Failed to send Telegram notification:", telegramError);
    }
  
    while (isAutoSellRunning) {
      try {
        let portfolio = {};
        const shortWallet = ${PUBLIC_KEY.slice(0, 4)}...${PUBLIC_KEY.slice(-4)};
        try {
          const data = fs.readFileSync(`portfolio_${shortWallet}.json`, "utf8");
          portfolio = JSON.parse(data);
        } catch (err) {
          // File doesn't exist or is invalid, start with empty portfolio
        }
  
        for (const [mint, details] of Object.entries(portfolio)) {
          if (details.amount > 0) {
            try {
              const price = await getPriceForMint(mint);
              if (price) {
                // console.log(`Current price for ${mint}: ${price} SOL`);
                await sellToken({ address: mint, ...details }, chat_id);
                // await monitoring_buy_one({ address: mint, ...details }, chat_id);
              } else {
                console.log(`❌ Failed to get price for ${mint}`);
                if (isPriceStreamRunning) {
                  console.log("🔄 Stopping price stream...");
                  stopPriceStream();
                }
                const pairData = await gettokeninfo(mint);
                if (pairData) {
                  console.log(`✅ New pair address found for ${mint}: ${pairData.pairAddress}`);
                  details.pairAddress = pairData.pairAddress;
                  details.liquidity = pairData.liquidity;
                  fs.writeFileSync(`portfolio_${shortWallet}.json`, JSON.stringify(portfolio, null, 2));
                } else {
                  console.log(`❌ Failed to get pair data for ${mint}`);
                }
                console.log("🔄 Starting price stream...");
                priceStreamPromise();
                await sleep(1000);
              }
            } catch (error) {
              console.error(`❌ Error processing token ${mint}:`, error);
              await sleep(1000);
            }
          }
        }
  
        await sleep(100); // Slightly longer delay to reduce load
      } catch (error) {
        console.error("Error monitoring positions:", error);
        await sleep(1000); // Wait before retrying on error
      }
    }
  
    console.log(chalk.red("Auto trading stopped"));
  };
  
  export const monitoring_buy_all = async (chat_id) => {
    try {
      while (true) {
        const spl_tokens = await getSPLTokens(PUBLIC_KEY);
        await Promise.all(spl_tokens.map((token) => monitoring_buy_one(token.mint, chat_id)));
        console.log("✅ monitoring and buying all tokens.");
        await sleep(1000);
      }
    } catch (error) {
      console.error("Error in monitoring_buy_all:", error);
    }
  };
    