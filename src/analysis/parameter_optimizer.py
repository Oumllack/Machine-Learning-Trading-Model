"""
Parameter optimization script for the trading system
"""

import pandas as pd
import numpy as np
from itertools import product
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from trading_bot_simple import SimpleTradingBot
from data_collector import DataCollector
from technical_analysis import TechnicalAnalysis
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Parameter optimization for trading system
    """
    
    def __init__(self):
        self.collector = DataCollector()
        self.results = []
        self.best_params = {}
        
    def optimize_parameters(self, symbol: str, param_ranges: Dict, 
                          optimization_method: str = 'grid') -> Dict:
        """
        Optimize trading parameters using specified method
        
        Args:
            symbol: Stock symbol to optimize for
            param_ranges: Dictionary of parameter ranges to test
            optimization_method: 'grid', 'genetic', or 'bayesian'
            
        Returns:
            Dictionary with best parameters and performance metrics
        """
        logger.info(f"Starting parameter optimization for {symbol}")
        
        if optimization_method == 'grid':
            return self._grid_search(symbol, param_ranges)
        elif optimization_method == 'genetic':
            return self._genetic_algorithm(symbol, param_ranges)
        elif optimization_method == 'bayesian':
            return self._bayesian_optimization(symbol, param_ranges)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def _grid_search(self, symbol: str, param_ranges: Dict) -> Dict:
        """
        Perform grid search optimization
        """
        logger.info("Performing grid search optimization...")
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))
        
        best_result = None
        best_sharpe = -np.inf
        
        total_combinations = len(combinations)
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(combinations):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total_combinations} combinations tested")
            
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Test parameters
            result = self._test_parameters(symbol, params)
            
            if result and result['sharpe_ratio'] > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_result = result
                logger.info(f"New best Sharpe ratio: {best_sharpe:.4f}")
        
        logger.info(f"Grid search completed. Best Sharpe ratio: {best_sharpe:.4f}")
        return best_result
    
    def _genetic_algorithm(self, symbol: str, param_ranges: Dict) -> Dict:
        """
        Perform genetic algorithm optimization
        """
        logger.info("Performing genetic algorithm optimization...")
        
        # Genetic algorithm parameters
        population_size = 50
        generations = 20
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population
        population = self._initialize_population(param_ranges, population_size)
        
        best_individual = None
        best_fitness = -np.inf
        
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                result = self._test_parameters(symbol, individual)
                fitness = result['sharpe_ratio'] if result else -np.inf
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    logger.info(f"New best fitness: {best_fitness:.4f}")
            
            # Selection
            selected = self._selection(population, fitness_scores, population_size // 2)
            
            # Crossover
            offspring = self._crossover(selected, crossover_rate)
            
            # Mutation
            offspring = self._mutation(offspring, param_ranges, mutation_rate)
            
            # New population
            population = selected + offspring
        
        # Test best individual
        best_result = self._test_parameters(symbol, best_individual)
        logger.info(f"Genetic algorithm completed. Best fitness: {best_fitness:.4f}")
        return best_result
    
    def _bayesian_optimization(self, symbol: str, param_ranges: Dict) -> Dict:
        """
        Perform Bayesian optimization
        """
        logger.info("Performing Bayesian optimization...")
        
        # Simple Bayesian optimization implementation
        n_iterations = 50
        best_params = None
        best_score = -np.inf
        
        for i in range(n_iterations):
            logger.info(f"Iteration {i + 1}/{n_iterations}")
            
            # Generate random parameters
            params = self._random_parameters(param_ranges)
            
            # Test parameters
            result = self._test_parameters(symbol, params)
            
            if result and result['sharpe_ratio'] > best_score:
                best_score = result['sharpe_ratio']
                best_params = params.copy()
                logger.info(f"New best score: {best_score:.4f}")
        
        # Test best parameters
        best_result = self._test_parameters(symbol, best_params)
        logger.info(f"Bayesian optimization completed. Best score: {best_score:.4f}")
        return best_result
    
    def _test_parameters(self, symbol: str, params: Dict) -> Dict:
        """
        Test a set of parameters and return performance metrics
        """
        try:
            # Create bot with custom parameters
            bot = SimpleTradingBot(symbol, config.INITIAL_CAPITAL)
            
            # Apply parameters
            bot.confidence_threshold = params.get('confidence_threshold', 0.6)
            bot.stop_loss_pct = params.get('stop_loss_pct', 0.05)
            bot.take_profit_pct = params.get('take_profit_pct', 0.15)
            
            # Run trading session
            bot.run_trading_session(days=60)
            
            # Get performance metrics
            metrics = bot.get_performance_metrics()
            
            if metrics['total_trades'] == 0:
                return None
            
            # Calculate Sharpe ratio
            returns = np.array(bot.daily_returns) if bot.daily_returns else np.array([0])
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            return {
                'params': params,
                'total_return_pct': metrics['total_return_pct'],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'profit_factor': metrics.get('profit_factor', 0)
            }
            
        except Exception as e:
            logger.error(f"Error testing parameters: {str(e)}")
            return None
    
    def _initialize_population(self, param_ranges: Dict, size: int) -> List[Dict]:
        """
        Initialize population for genetic algorithm
        """
        population = []
        for _ in range(size):
            individual = self._random_parameters(param_ranges)
            population.append(individual)
        return population
    
    def _random_parameters(self, param_ranges: Dict) -> Dict:
        """
        Generate random parameters within specified ranges
        """
        params = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, (list, tuple)):
                params[param_name] = np.random.choice(param_range)
            else:
                params[param_name] = param_range
        return params
    
    def _selection(self, population: List[Dict], fitness_scores: List[float], 
                  n_select: int) -> List[Dict]:
        """
        Tournament selection for genetic algorithm
        """
        selected = []
        for _ in range(n_select):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected
    
    def _crossover(self, selected: List[Dict], crossover_rate: float) -> List[Dict]:
        """
        Crossover operation for genetic algorithm
        """
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected) and np.random.random() < crossover_rate:
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                # Single point crossover
                child1 = parent1.copy()
                child2 = parent2.copy()
                
                # Randomly swap some parameters
                for key in parent1.keys():
                    if np.random.random() < 0.5:
                        child1[key], child2[key] = child2[key], child1[key]
                
                offspring.extend([child1, child2])
            else:
                offspring.append(selected[i].copy())
        
        return offspring
    
    def _mutation(self, offspring: List[Dict], param_ranges: Dict, 
                  mutation_rate: float) -> List[Dict]:
        """
        Mutation operation for genetic algorithm
        """
        for individual in offspring:
            for param_name, param_range in param_ranges.items():
                if np.random.random() < mutation_rate:
                    if isinstance(param_range, (list, tuple)):
                        individual[param_name] = np.random.choice(param_range)
                    else:
                        individual[param_name] = param_range
        return offspring
    
    def optimize_multiple_stocks(self, symbols: List[str], 
                               param_ranges: Dict) -> Dict:
        """
        Optimize parameters for multiple stocks
        """
        logger.info(f"Optimizing parameters for {len(symbols)} stocks...")
        
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"Optimizing {symbol}...")
            result = self.optimize_parameters(symbol, param_ranges)
            all_results[symbol] = result
        
        # Find best overall parameters
        best_overall = self._find_best_overall_params(all_results)
        
        return {
            'individual_results': all_results,
            'best_overall_params': best_overall
        }
    
    def _find_best_overall_params(self, results: Dict) -> Dict:
        """
        Find best overall parameters across all stocks
        """
        # Calculate average performance for each parameter set
        param_performance = {}
        
        for symbol, result in results.items():
            if result:
                param_key = str(result['params'])
                if param_key not in param_performance:
                    param_performance[param_key] = {
                        'params': result['params'],
                        'sharpe_ratios': [],
                        'total_returns': [],
                        'win_rates': []
                    }
                
                param_performance[param_key]['sharpe_ratios'].append(result['sharpe_ratio'])
                param_performance[param_key]['total_returns'].append(result['total_return_pct'])
                param_performance[param_key]['win_rates'].append(result['win_rate'])
        
        # Find best average performance
        best_params = None
        best_avg_sharpe = -np.inf
        
        for param_key, performance in param_performance.items():
            avg_sharpe = np.mean(performance['sharpe_ratios'])
            if avg_sharpe > best_avg_sharpe:
                best_avg_sharpe = avg_sharpe
                best_params = performance['params']
        
        return {
            'params': best_params,
            'avg_sharpe_ratio': best_avg_sharpe,
            'performance_summary': param_performance
        }

def main():
    """
    Run parameter optimization
    """
    optimizer = ParameterOptimizer()
    
    # Define parameter ranges to test
    param_ranges = {
        'confidence_threshold': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'stop_loss_pct': [0.02, 0.03, 0.05, 0.07, 0.10],
        'take_profit_pct': [0.05, 0.10, 0.15, 0.20, 0.25]
    }
    
    # Test stocks
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    # Run optimization
    results = optimizer.optimize_multiple_stocks(symbols, param_ranges)
    
    # Print results
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("="*60)
    
    for symbol, result in results['individual_results'].items():
        print(f"\n{symbol}:")
        if result:
            print(f"  Best Parameters: {result['params']}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"  Total Return: {result['total_return_pct']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.2%}")
            print(f"  Total Trades: {result['total_trades']}")
        else:
            print("  No valid results found")
    
    print(f"\nBest Overall Parameters:")
    best_overall = results['best_overall_params']
    print(f"  Parameters: {best_overall['params']}")
    print(f"  Average Sharpe Ratio: {best_overall['avg_sharpe_ratio']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimization_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, default=convert_numpy, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main() 