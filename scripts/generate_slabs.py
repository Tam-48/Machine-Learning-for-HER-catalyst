"""
Slab generation script for bimetallic catalysts.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import Config, get_logger
from src.slab_generation import SlabBuilder

logger = get_logger(__name__)


def main():
    """Generate slab structures."""
    
    logger.info("=" * 60)
    logger.info("Bimetallic Slab Generation")
    logger.info("=" * 60)
    
    # Load configuration
    config = Config()
    logger.info("Configuration loaded")
    
    # Initialize slab builder
    logger.info("\n[Step 1] Initializing slab builder...")
    builder = SlabBuilder(config)
    
    # Define metal pairs to generate (from Stage 1 top results)
    top_pairs = [
        {'metal_base': 'Ni', 'metal_a': 'Fe', 'metal_b': 'Co', 'ratio': 0.5},
        {'metal_base': 'Ni', 'metal_a': 'Mo', 'metal_b': 'W', 'ratio': 0.5},
        {'metal_base': 'Pt', 'metal_a': 'Pd', 'metal_b': 'Ru', 'ratio': 0.5},
    ]
    
    # Step 2: Generate structures
    logger.info("\n[Step 2] Generating structures...")
    
    for pair in top_pairs:
        logger.info(f"\nGenerating {pair['metal_a']}-{pair['metal_b']} "
                   f"on {pair['metal_base']} substrate")
        
        # Generate composition
        slabs = builder.generate_composition_series(
            metal_base=pair['metal_base'],
            metal_a=pair['metal_a'],
            metal_b=pair['metal_b'],
            ratio=pair['ratio']
        )
        
        # Save structures
        output_dir = f"data/structures/{pair['metal_a']}_{pair['metal_b']}/"
        builder.save_structures(slabs, format='extxyz', output_dir=output_dir)
        logger.info(f"Saved to {output_dir}")
    
    # Step 3: Generate ratio series for top candidate
    logger.info("\n[Step 3] Generating ratio series...")
    
    best_pair = top_pairs[0]
    ratio_slabs = builder.generate_ratio_series(
        metal_base=best_pair['metal_base'],
        metal_a=best_pair['metal_a'],
        metal_b=best_pair['metal_b'],
        x_values=None  # Uses default 0.0-1.0 at 0.1 intervals
    )
    
    ratio_dir = f"data/structures/{best_pair['metal_a']}_{best_pair['metal_b']}_ratio/"
    builder.save_structures(ratio_slabs, format='extxyz', output_dir=ratio_dir)
    logger.info(f"Generated {len(ratio_slabs)} ratio variations")
    logger.info(f"Saved to {ratio_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Slab Generation Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
