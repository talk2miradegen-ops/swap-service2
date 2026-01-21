# Railway main.py - v3.0.0 with imported wallet support

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import httpx
import asyncio
import base64
import base58
import os
import time
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.signature import Signature
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TxOpts
import hashlib

# ============================================================
#                     CONFIGURATION
# ============================================================

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "change-this-key")
MASTER_SEED_PHRASE = os.getenv("MASTER_SEED_PHRASE", "")

QUICKNODE_RPC = os.getenv(
    "QUICKNODE_RPC",
    "https://old-dawn-hexagon.solana-mainnet.quiknode.pro/879214d71e34c1e59d00dc0a8c62319ea5916f9c"
)

RPC_URL = QUICKNODE_RPC
WSOL_MINT = "So11111111111111111111111111111111111111112"

TOKEN_ACCOUNT_RENT = 0.00205
ATA_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

JUPITER_ENDPOINTS = [
    {"name": "Jupiter Main", "quote": "https://quote-api.jup.ag/v6/quote", "swap": "https://quote-api.jup.ag/v6/swap"},
    {"name": "Jupiter Public", "quote": "https://public.jupiterapi.com/quote", "swap": "https://public.jupiterapi.com/swap"},
    {"name": "Jupiter Lite", "quote": "https://lite-api.jup.ag/v6/quote", "swap": "https://lite-api.jup.ag/v6/swap"},
]

MAX_RETRIES = 3
RETRY_DELAY = 1.5

# ============================================================
#                     FASTAPI APP
# ============================================================

app = FastAPI(title="Solana Swap Service", version="3.0.0")

# ============================================================
#                     MODELS (UPDATED!)
# ============================================================

class SwapRequest(BaseModel):
    wallet_index: int
    token_address: str
    amount: float
    slippage: float = 50.0
    priority_fee: float = 0.00005
    # NEW: Optional private key for imported wallets
    private_key: Optional[str] = None

class SwapResponse(BaseModel):
    success: bool
    message: str
    signature: Optional[str] = None
    in_amount: Optional[str] = None
    out_amount: Optional[str] = None
    price_impact: Optional[str] = None
    dex_used: Optional[str] = None
    fees_paid: Optional[str] = None
    retries: Optional[int] = None

# ============================================================
#                     WALLET (UPDATED!)
# ============================================================

def derive_keypair(index: int) -> Keypair:
    """Derive keypair from master seed"""
    if not MASTER_SEED_PHRASE:
        raise ValueError("MASTER_SEED_PHRASE not set")
    
    mnemonic = MASTER_SEED_PHRASE.strip()
    salt = b"mnemonic"
    master_seed = hashlib.pbkdf2_hmac('sha512', mnemonic.encode(), salt, 2048)
    unique_data = master_seed + index.to_bytes(4, 'little')
    derived_hash = hashlib.sha256(unique_data).digest()
    
    return Keypair.from_seed(derived_hash)


def keypair_from_private_key(private_key_b58: str) -> Keypair:
    """Create keypair from base58 private key"""
    try:
        private_key_bytes = base58.b58decode(private_key_b58.strip())
        
        if len(private_key_bytes) == 64:
            return Keypair.from_bytes(private_key_bytes)
        elif len(private_key_bytes) == 32:
            return Keypair.from_seed(private_key_bytes)
        else:
            raise ValueError(f"Invalid private key length: {len(private_key_bytes)}")
    except Exception as e:
        raise ValueError(f"Invalid private key: {e}")


def get_keypair(wallet_index: int, private_key: Optional[str] = None) -> Keypair:
    """Get keypair - supports both derived and imported wallets"""
    
    # If private key provided, use it (imported wallet)
    if private_key:
        print(f"   🔑 Using imported wallet")
        return keypair_from_private_key(private_key)
    
    # Otherwise, derive from master seed
    if wallet_index < 0:
        raise ValueError("wallet_index < 0 requires private_key")
    
    print(f"   🔑 Using derived wallet (index: {wallet_index})")
    return derive_keypair(wallet_index)

# ============================================================
#                     HTTP WITH DNS FAILOVER
# ============================================================

async def robust_http_get(endpoints: list, endpoint_key: str, params: dict) -> tuple:
    errors = []
    
    for ep in endpoints:
        url = ep[endpoint_key]
        ep_name = ep["name"]
        
        print(f"   🔗 Trying {ep_name}...")
        
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                    resp = await client.get(url, params=params)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if "outAmount" in data:
                            print(f"   ✅ {ep_name} success")
                            return data, ep_name
                        elif "error" in data:
                            errors.append(f"{ep_name}: {data['error']}")
                            break
                    else:
                        errors.append(f"{ep_name}: HTTP {resp.status_code}")
                        break
                        
            except httpx.ConnectError:
                errors.append(f"{ep_name}: Connection error")
                await asyncio.sleep(0.5)
            except httpx.TimeoutException:
                errors.append(f"{ep_name}: Timeout")
                await asyncio.sleep(0.5)
            except Exception as e:
                errors.append(f"{ep_name}: {str(e)[:50]}")
                break
        
        await asyncio.sleep(0.3)
    
    raise Exception(f"All endpoints failed: {'; '.join(errors[-3:])}")


async def robust_http_post(endpoints: list, endpoint_key: str, data: dict, preferred: str = None) -> tuple:
    sorted_eps = sorted(endpoints, key=lambda x: 0 if x["name"] == preferred else 1)
    errors = []
    
    for ep in sorted_eps:
        url = ep[endpoint_key]
        ep_name = ep["name"]
        
        print(f"   🔗 POST to {ep_name}...")
        
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
                    resp = await client.post(url, json=data)
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        if "swapTransaction" in result:
                            print(f"   ✅ Got swap transaction")
                            return result["swapTransaction"], ep_name
                        elif "error" in result:
                            errors.append(f"{ep_name}: {result['error']}")
                            break
                    else:
                        errors.append(f"{ep_name}: HTTP {resp.status_code}")
                        break
                        
            except httpx.ConnectError:
                errors.append(f"{ep_name}: Connection error")
                await asyncio.sleep(0.5)
            except httpx.TimeoutException:
                errors.append(f"{ep_name}: Timeout")
                await asyncio.sleep(0.5)
            except Exception as e:
                errors.append(f"{ep_name}: {str(e)[:50]}")
                break
        
        await asyncio.sleep(0.3)
    
    raise Exception(f"All endpoints failed: {'; '.join(errors[-3:])}")

# ============================================================
#                     BALANCE & TOKEN ACCOUNT
# ============================================================

async def get_sol_balance(rpc: AsyncClient, wallet: str) -> float:
    try:
        pubkey = Pubkey.from_string(wallet)
        response = await rpc.get_balance(pubkey)
        return response.value / 1e9
    except Exception as e:
        print(f"   ⚠️ Balance error: {e}")
        return 0


def get_associated_token_address(wallet: str, mint: str) -> Pubkey:
    wallet_pubkey = Pubkey.from_string(wallet)
    mint_pubkey = Pubkey.from_string(mint)
    token_program = Pubkey.from_string(TOKEN_PROGRAM_ID)
    ata_program = Pubkey.from_string(ATA_PROGRAM_ID)
    
    seeds = [bytes(wallet_pubkey), bytes(token_program), bytes(mint_pubkey)]
    ata, _ = Pubkey.find_program_address(seeds, ata_program)
    return ata


async def check_token_account_exists(rpc: AsyncClient, wallet: str, mint: str) -> bool:
    try:
        ata = get_associated_token_address(wallet, mint)
        response = await rpc.get_account_info(ata)
        return response.value is not None
    except:
        return False


async def calculate_fees(rpc: AsyncClient, wallet: str, mint: str, 
                         priority_fee: float, is_buy: bool) -> dict:
    fees = {
        "rent": 0.0,
        "priority": priority_fee,
        "base": 0.000005,
        "buffer": 0.0002,
    }
    
    if is_buy:
        has_account = await check_token_account_exists(rpc, wallet, mint)
        if not has_account:
            fees["rent"] = TOKEN_ACCOUNT_RENT
            print(f"   📋 Need token account: +{TOKEN_ACCOUNT_RENT} SOL")
        else:
            print(f"   ✅ Token account exists")
    
    fees["total"] = fees["rent"] + fees["priority"] + fees["base"] + fees["buffer"]
    return fees

# ============================================================
#                     JUPITER SWAP
# ============================================================

async def get_quote(input_mint: str, output_mint: str, 
                    amount_lamports: int, slippage_bps: int) -> tuple:
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": str(slippage_bps),
    }
    
    print(f"   📊 Quote: {amount_lamports} lamports, {slippage_bps} bps")
    quote, source = await robust_http_get(JUPITER_ENDPOINTS, "quote", params)
    print(f"   ✅ Out: {quote.get('outAmount', '?')}")
    
    return quote, source


async def build_swap_tx(quote: dict, public_key: str, 
                        priority_lamports: int, preferred: str) -> tuple:
    payload = {
        "quoteResponse": quote,
        "userPublicKey": public_key,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": priority_lamports,
    }
    
    return await robust_http_post(JUPITER_ENDPOINTS, "swap", payload, preferred)


async def execute_swap_with_retry(rpc: AsyncClient, keypair: Keypair,
                                   input_mint: str, output_mint: str,
                                   amount_lamports: int, slippage_bps: int,
                                   priority_lamports: int) -> dict:
    public_key = str(keypair.pubkey())
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"\n   🔄 Attempt {attempt + 1}/{MAX_RETRIES}")
            
            quote, quote_source = await get_quote(
                input_mint, output_mint, amount_lamports, slippage_bps
            )
            
            await asyncio.sleep(0.2)
            
            swap_tx, swap_source = await build_swap_tx(
                quote, public_key, priority_lamports, quote_source
            )
            
            tx_bytes = base64.b64decode(swap_tx)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(transaction.message, [keypair])
            print(f"   ✍️ Signed")
            
            opts = TxOpts(skip_preflight=True, preflight_commitment=Processed)
            result = await rpc.send_transaction(signed_tx, opts)
            
            signature = str(result.value).strip()
            print(f"   📤 Sent: {signature[:30]}...")
            
            await asyncio.sleep(2)
            sig_obj = Signature.from_string(signature)
            
            for check in range(20):
                status = await rpc.get_signature_statuses([sig_obj])
                
                if status.value and status.value[0]:
                    stat = status.value[0]
                    
                    if stat.err:
                        error_str = str(stat.err)
                        print(f"   ❌ Error: {error_str[:50]}")
                        
                        if "6014" in error_str:
                            last_error = "Slippage exceeded"
                            break
                        last_error = error_str
                        break
                    
                    if stat.confirmation_status:
                        print(f"   ✅ Confirmed!")
                        return {
                            "success": True,
                            "signature": signature,
                            "in_amount": quote.get("inAmount"),
                            "out_amount": quote.get("outAmount"),
                            "price_impact": quote.get("priceImpactPct", "0"),
                            "retries": attempt
                        }
                
                await asyncio.sleep(2)
            
            if last_error is None:
                return {
                    "success": True,
                    "signature": signature,
                    "in_amount": quote.get("inAmount"),
                    "out_amount": quote.get("outAmount"),
                    "price_impact": quote.get("priceImpactPct", "0"),
                    "retries": attempt,
                }
                
        except Exception as e:
            last_error = str(e)
            print(f"   ❌ Error: {last_error[:60]}")
        
        if attempt < MAX_RETRIES - 1:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"   ⏳ Retry in {wait}s...")
            await asyncio.sleep(wait)
    
    raise Exception(f"Failed after {MAX_RETRIES} attempts: {last_error}")

# ============================================================
#                     MAIN SWAP (UPDATED!)
# ============================================================

async def execute_swap(wallet_index: int, input_mint: str, output_mint: str,
                       amount: float, slippage: float, priority_fee: float,
                       private_key: Optional[str] = None) -> dict:
    
    print(f"\n{'='*60}")
    print(f"🚀 SWAP v3.0.0")
    print(f"{'='*60}")
    
    # Get keypair - works for both derived and imported wallets
    keypair = get_keypair(wallet_index, private_key)
    public_key = str(keypair.pubkey())
    
    is_buy = input_mint == WSOL_MINT
    target_token = output_mint if is_buy else input_mint
    
    print(f"   {'BUY' if is_buy else 'SELL'}")
    print(f"   Wallet: {public_key[:20]}...")
    print(f"   Amount: {amount:.8f}")
    print(f"   Slippage: {slippage}%")
    
    rpc = AsyncClient(RPC_URL)
    
    try:
        balance = await get_sol_balance(rpc, public_key)
        print(f"\n💰 Balance: {balance:.6f} SOL")
        
        fees = await calculate_fees(rpc, public_key, target_token, priority_fee, is_buy)
        print(f"📊 Fees: {fees['total']:.5f} SOL")
        
        if is_buy:
            available = balance - fees['total']
            print(f"💵 Available: {available:.6f} SOL")
            
            if available <= 0.0001:
                raise HTTPException(400, f"Insufficient balance: {balance:.4f} SOL")
            
            if amount > available:
                print(f"   ⚠️ Adjusting: {amount:.6f} → {available:.6f}")
                amount = available
        
        amount_lamports = int(amount * 1e9)
        slippage_bps = int(slippage * 100)
        priority_lamports = int(priority_fee * 1e9)
        
        if amount_lamports < 100000:
            raise HTTPException(400, "Amount too small")
        
        result = await execute_swap_with_retry(
            rpc, keypair, input_mint, output_mint,
            amount_lamports, slippage_bps, priority_lamports
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"   🔗 https://solscan.io/tx/{result['signature']}")
        
        result["dex_used"] = "Jupiter v6"
        result["fees_paid"] = f"{fees['total']:.5f}"
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise HTTPException(500, str(e))
    finally:
        await rpc.close()

# ============================================================
#                     ENDPOINTS (UPDATED!)
# ============================================================

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(401, "Invalid API key")


@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0", "features": ["imported_wallets"]}


@app.post("/api/swap/buy", response_model=SwapResponse)
async def buy(req: SwapRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    
    print(f"\n{'#'*50}")
    print(f"💰 BUY: {req.amount} SOL → {req.token_address[:15]}...")
    print(f"   Wallet Index: {req.wallet_index}")
    print(f"   Has Private Key: {'Yes' if req.private_key else 'No'}")
    print(f"{'#'*50}")
    
    try:
        result = await execute_swap(
            req.wallet_index, WSOL_MINT, req.token_address,
            req.amount, req.slippage, req.priority_fee,
            req.private_key  # Pass private key for imported wallets
        )
        return SwapResponse(success=True, message="OK", **{
            k: result.get(k) for k in 
            ["signature", "in_amount", "out_amount", "price_impact", "dex_used", "fees_paid", "retries"]
        })
    except HTTPException as e:
        return SwapResponse(success=False, message=e.detail)
    except Exception as e:
        return SwapResponse(success=False, message=str(e))


@app.post("/api/swap/sell", response_model=SwapResponse)
async def sell(req: SwapRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    
    print(f"\n{'#'*50}")
    print(f"💸 SELL: {req.amount} {req.token_address[:15]}... → SOL")
    print(f"   Wallet Index: {req.wallet_index}")
    print(f"   Has Private Key: {'Yes' if req.private_key else 'No'}")
    print(f"{'#'*50}")
    
    try:
        result = await execute_swap(
            req.wallet_index, req.token_address, WSOL_MINT,
            req.amount, req.slippage, req.priority_fee,
            req.private_key  # Pass private key for imported wallets
        )
        return SwapResponse(success=True, message="OK", **{
            k: result.get(k) for k in 
            ["signature", "in_amount", "out_amount", "price_impact", "dex_used", "fees_paid", "retries"]
        })
    except HTTPException as e:
        return SwapResponse(success=False, message=e.detail)
    except Exception as e:
        return SwapResponse(success=False, message=str(e))


@app.get("/api/wallet/{index}")
async def get_wallet(index: int, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    return {"public_key": str(derive_keypair(index).pubkey())}


@app.get("/api/balance/{wallet}")
async def get_balance(wallet: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    rpc = AsyncClient(RPC_URL)
    try:
        return {"balance_sol": await get_sol_balance(rpc, wallet)}
    finally:
        await rpc.close()


@app.on_event("startup")
async def startup():
    print("=" * 50)
    print("🚀 SWAP v3.0.0 - Imported Wallet Support")
    print("=" * 50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
