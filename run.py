#!/usr/bin/env python3
"""
Main entry point for the Advanced Trading System
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Main entry point with command line interface
    """
    parser = argparse.ArgumentParser(description='Advanced Trading System')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--symbol', type=str, 
                       help='Stock symbol to analyze')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to simulate (default: 30)')
    parser.add_argument('--portfolio', type=str,
                       help='Comma-separated list of symbols for portfolio simulation')
    parser.add_argument('--aggressive', action='store_true',
                       help='Use aggressive trading parameters')
    parser.add_argument('--analyze', action='store_true',
                       help='Generate comprehensive analysis')
    parser.add_argument('--demo', action='store_true',
                       help='Run optimized demo')
    
    args = parser.parse_args()
    
    if args.interactive:
        from src.trading_system import TradingSystem
        system = TradingSystem()
        system.interactive_mode()
    
    elif args.analyze:
        from src.analysis.generate_analysis_simple import main as run_analysis
        run_analysis()
    
    elif args.demo:
        from src.demos.demo_optimized import main as run_demo
        run_demo()
    
    elif args.portfolio:
        from src.trading_system import TradingSystem
        system = TradingSystem()
        symbols = [s.strip() for s in args.portfolio.split(',')]
        system.run_portfolio_simulation(
            symbols=symbols,
            capital_per_symbol=args.capital,
            days=args.days,
            aggressive=args.aggressive
        )
    
    elif args.symbol:
        from src.trading_system import TradingSystem
        system = TradingSystem()
        
        # Run analysis
        system.run_analysis(args.symbol)
        
        # Run trading session
        system.run_trading_session(
            symbol=args.symbol,
            days=args.days,
            capital=args.capital,
            aggressive=args.aggressive
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 