# ================== FILE: models/blockchain.py ==================

import time
import hashlib
import numpy as np


class MockPoABlockchain:
    """Mock PoA Blockchain for simulation"""

    def __init__(self):
        self.blocks = []
        self.validators = ['V1', 'V2', 'V3']
        self.block_time = 5  # seconds
        self.energy_per_tx = 0.1  # kWh

    def create_tx(self, user_id, payment, lot_bonus=0, ref_bonus=0,
                  verify_conditions=None):
        """Create transaction (smart contract simulation)"""
        if verify_conditions:
            if not all(verify_conditions.values()):
                print(f"❌ TX rejected for user {user_id}: Verify failed")
                return None

        tx = {
            'user_id': user_id,
            'payment': payment,
            'lot': lot_bonus,
            'ref': ref_bonus,
            'timestamp': time.time()
        }
        tx_hash = hashlib.sha256(str(tx).encode()).hexdigest()[:10]
        print(f"📝 Created TX {tx_hash} for user {user_id}")
        return {'hash': tx_hash, 'data': tx}

    def mine_block(self, txs):
        """PoA consensus: Random validator approves"""
        if not txs:
            return

        prev_hash = self.blocks[-1]['hash'] if self.blocks else 'genesis'
        validator = np.random.choice(self.validators)
        energy = len(txs) * self.energy_per_tx

        block = {
            'index': len(self.blocks) + 1,
            'txs': txs,
            'prev_hash': prev_hash,
            'validator': validator,
            'energy': energy
        }
        block['hash'] = hashlib.sha256(str(block).encode()).hexdigest()[:10]
        self.blocks.append(block)
        print(f"🔗 Mined block {block['index']}: {len(txs)} txs")
