"""
Main Entry Point: MARL CTDE Industrial Demo Suite

Run this script to execute complete demonstrations of:
1. Retail: Multi-warehouse inventory optimization (MADDPG)
2. Banking: Transaction routing optimization (MAPPO)

Usage:
    python main.py --retail      # Run retail demo only
    python main.py --banking     # Run banking demo only
    python main.py --all         # Run all demos
    python main.py --help        # Show help
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MARL CTDE Industrial Demo Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --retail      # Run retail inventory demo
  python main.py --banking     # Run banking transaction routing demo
  python main.py --all         # Run all demos
  python main.py --list        # List available demos
        """
    )
    
    parser.add_argument('--retail', action='store_true', 
                       help='Run retail inventory optimization demo')
    parser.add_argument('--banking', action='store_true',
                       help='Run banking transaction routing demo')
    parser.add_argument('--all', action='store_true',
                       help='Run all demos')
    parser.add_argument('--list', action='store_true',
                       help='List available demos')
    
    args = parser.parse_args()
    
    # Show list of demos
    if args.list:
        print("\nAvailable Demonstrations:")
        print("-" * 60)
        print("1. RETAIL (CTDE-MADDPG)")
        print("   Multi-warehouse inventory optimization")
        print("   Algorithm: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)")
        print("   Use case: Supply chain coordination")
        print()
        print("2. BANKING (CTDE-MAPPO)")
        print("   Transaction routing across channels")
        print("   Algorithm: MAPPO (Multi-Agent Proximal Policy Optimization)")
        print("   Use case: Financial transaction management")
        print()
        print("Run with: python main.py --retail|--banking|--all")
        return
    
    # Determine which demos to run
    run_retail = args.retail or args.all
    run_banking = args.banking or args.all
    
    # If none specified, show help
    if not run_retail and not run_banking:
        parser.print_help()
        return
    
    # Print welcome message
    print("\n" + "="*80)
    print(" MULTI-AGENT REINFORCEMENT LEARNING (MARL) CTDE INDUSTRIAL DEMO SUITE")
    print("="*80)
    print("\nThis demo suite showcases Centralized Training with Decentralized Execution")
    print("(CTDE) pattern in real-world industrial applications.\n")
    
    # Run retail demo
    if run_retail:
        print("\n" + "="*80)
        print(" STARTING RETAIL DEMO (CTDE-MADDPG)")
        print("="*80)
        try:
            from retail.demo import demo_train_and_evaluate
            demo_train_and_evaluate()
        except Exception as e:
            print(f"Error running retail demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Run banking demo
    if run_banking:
        print("\n" + "="*80)
        print(" STARTING BANKING DEMO (CTDE-MAPPO)")
        print("="*80)
        try:
            from banking.demo import main as banking_main
            banking_main()
        except Exception as e:
            print(f"Error running banking demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print(" ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. CTDE enables coordination without explicit communication")
    print("2. Different algorithms (MADDPG, MAPPO) suit different problems")
    print("3. Centralized training with decentralized execution scales better")
    print("4. Multi-agent systems can solve complex coordination problems")
    print("\nFor more information, see README.md and docs/ folder")
    print("\n")


if __name__ == "__main__":
    main()
