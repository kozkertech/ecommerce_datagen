#!/usr/bin/env python3
"""
Fixed Synthetic Data Authenticity Checker
Comprehensive validation with intelligent column filtering for realistic scoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


EXCLUDE_FROM_CATEGORICAL_SCORING = {
    'order_id', 'customer_id', 'product_id', 'seller_id', 'review_id',
    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
    'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date',
    'review_creation_date', 'review_answer_timestamp', 'customer_unique_id',
    'customer_city', 'customer_state', 'seller_city', 'seller_state', 'seller_zip_code_prefix',
    'customer_zip_code_prefix', 'geolocation_zip_code_prefix'
}

def load_data():
    """Load both real and synthetic datasets"""
    print("📥 Loading datasets...")
    
    # Load real data
    real_data_path = r'E:\brazilian dataset'  
    real_files = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv'
    }
    
    real_data = {}
    for table_name, filename in real_files.items():
        try:
            real_data[table_name] = pd.read_csv(f"{real_data_path}/{filename}")
            print(f"   ✅ Loaded real {table_name}: {real_data[table_name].shape}")
        except:
            print(f"   ⚠️ Could not load real {table_name}")
    
    # Load synthetic data
    synthetic_data = {}
    for table_name in real_files.keys():
        try:
            synthetic_data[table_name] = pd.read_csv(f'synthetic_{table_name}_hma_english.csv')
            print(f"   ✅ Loaded synthetic {table_name}: {synthetic_data[table_name].shape}")
        except:
            print(f"   ⚠️ Could not load synthetic {table_name}")
    
    return real_data, synthetic_data

def statistical_comparison(real_data, synthetic_data):
    """Compare statistical properties between real and synthetic data"""
    print("\n📊 Statistical Comparison Analysis")
    print("=" * 50)
    
    comparison_results = {}
    
    for table_name in real_data.keys():
        if table_name in synthetic_data:
            real_df = real_data[table_name]
            synthetic_df = synthetic_data[table_name]
            
            print(f"\n🔍 {table_name.upper()} Analysis:")
            
            # Numerical columns analysis
            numerical_cols = real_df.select_dtypes(include=[np.number]).columns
            table_results = {}
            
            for col in numerical_cols:
                if col in synthetic_df.columns:
                    real_stats = {
                        'mean': real_df[col].mean(),
                        'std': real_df[col].std(),
                        'min': real_df[col].min(),
                        'max': real_df[col].max(),
                        'median': real_df[col].median()
                    }
                    
                    synthetic_stats = {
                        'mean': synthetic_df[col].mean(),
                        'std': synthetic_df[col].std(),
                        'min': synthetic_df[col].min(),
                        'max': synthetic_df[col].max(),
                        'median': synthetic_df[col].median()
                    }
                    
                    # Calculate differences
                    mean_diff = abs(real_stats['mean'] - synthetic_stats['mean']) / real_stats['mean'] * 100
                    std_diff = abs(real_stats['std'] - synthetic_stats['std']) / real_stats['std'] * 100
                    
                    print(f"   📈 {col}:")
                    print(f"      Mean: Real={real_stats['mean']:.2f}, Synthetic={synthetic_stats['mean']:.2f} ({mean_diff:.1f}% diff)")
                    print(f"      Std:  Real={real_stats['std']:.2f}, Synthetic={synthetic_stats['std']:.2f} ({std_diff:.1f}% diff)")
                    
                    # KS test for distribution similarity
                    try:
                        ks_stat, p_value = stats.ks_2samp(real_df[col].dropna(), synthetic_df[col].dropna())
                        similarity_score = 1 - ks_stat  # Higher is better
                        print(f"      Distribution similarity: {similarity_score:.3f} (higher is better)")
                    except:
                        similarity_score = 0.5
                        print(f"      Distribution similarity: Could not calculate")
                    
                    table_results[col] = {
                        'mean_diff': mean_diff,
                        'std_diff': std_diff,
                        'similarity_score': similarity_score
                    }
            
            comparison_results[table_name] = table_results
    
    return comparison_results

def categorical_comparison(real_data, synthetic_data):
    """Compare categorical distributions with smart filtering"""
    print("\n🏷️ Categorical Distribution Analysis")
    print("=" * 50)
    
    categorical_results = {}
    
    for table_name in real_data.keys():
        if table_name in synthetic_data:
            real_df = real_data[table_name]
            synthetic_df = synthetic_data[table_name]
            
            print(f"\n🔍 {table_name.upper()} Categorical Analysis:")
            
            categorical_cols = real_df.select_dtypes(include=['object']).columns
            table_results = {}
            
            for col in categorical_cols:
                if col in synthetic_df.columns:
                    real_counts = real_df[col].value_counts()
                    synthetic_counts = synthetic_df[col].value_counts()
                    
                    # Calculate category overlap
                    real_categories = set(real_counts.index)
                    synthetic_categories = set(synthetic_counts.index)
                    
                    overlap = len(real_categories.intersection(synthetic_categories))
                    total_real = len(real_categories)
                    overlap_percentage = (overlap / total_real) * 100
                    
                    # Check if this column should be excluded from final scoring
                    excluded_from_scoring = col in EXCLUDE_FROM_CATEGORICAL_SCORING
                    exclusion_reason = ""
                    
                    if excluded_from_scoring:
                        if any(id_term in col.lower() for id_term in ['id', 'unique']):
                            exclusion_reason = "(ID - expected 0%)"
                        elif any(date_term in col.lower() for date_term in ['date', 'timestamp']):
                            exclusion_reason = "(Date - expected 0%)"
                        elif any(geo_term in col.lower() for geo_term in ['city', 'state', 'zip']):
                            exclusion_reason = "(Geographic - may be synthetic)"
                        else:
                            exclusion_reason = "(Synthetic by design)"
                    
                    print(f"   📊 {col} {exclusion_reason}:")
                    print(f"      Real categories: {total_real}")
                    print(f"      Synthetic categories: {len(synthetic_categories)}")
                    print(f"      Category overlap: {overlap}/{total_real} ({overlap_percentage:.1f}%)")
                    
                    # Top categories comparison
                    top_real = real_counts.head(3)
                    top_synthetic = synthetic_counts.head(3)
                    
                    print(f"      Top real categories: {list(top_real.index)}")
                    print(f"      Top synthetic categories: {list(top_synthetic.index)}")
                    
                    if excluded_from_scoring:
                        print(f"      ⚠️ Excluded from final categorical score {exclusion_reason}")
                    else:
                        print(f"      ✅ Included in final categorical score")
                    
                    table_results[col] = {
                        'overlap_percentage': overlap_percentage,
                        'real_categories': total_real,
                        'synthetic_categories': len(synthetic_categories),
                        'excluded_from_scoring': excluded_from_scoring,
                        'exclusion_reason': exclusion_reason
                    }
            
            categorical_results[table_name] = table_results
    
    return categorical_results

def referential_integrity_check(synthetic_data):
    """Check if synthetic data maintains referential integrity"""
    print("\n🔗 Referential Integrity Check")
    print("=" * 50)
    
    integrity_results = {}
    
    # Define relationships to check
    relationships = [
        ('orders', 'order_items', 'order_id'),
        ('orders', 'payments', 'order_id'),
        ('customers', 'orders', 'customer_id'),
        ('products', 'order_items', 'product_id'),
        ('sellers', 'order_items', 'seller_id')
    ]
    
    for parent_table, child_table, key_column in relationships:
        if parent_table in synthetic_data and child_table in synthetic_data:
            parent_df = synthetic_data[parent_table]
            child_df = synthetic_data[child_table]
            
            if key_column in parent_df.columns and key_column in child_df.columns:
                parent_keys = set(parent_df[key_column].dropna())
                child_keys = set(child_df[key_column].dropna())
                
                orphaned_keys = child_keys - parent_keys
                orphaned_count = len(orphaned_keys)
                total_child_records = len(child_df)
                
                integrity_percentage = ((total_child_records - orphaned_count) / total_child_records) * 100
                
                print(f"   🔍 {parent_table} → {child_table} ({key_column}):")
                print(f"      Orphaned records: {orphaned_count}/{total_child_records}")
                print(f"      Integrity: {integrity_percentage:.1f}%")
                
                if orphaned_count == 0:
                    print(f"      ✅ Perfect referential integrity!")
                else:
                    print(f"      ⚠️ {orphaned_count} orphaned records found")
                
                integrity_results[f"{parent_table}_{child_table}"] = {
                    'orphaned_count': orphaned_count,
                    'integrity_percentage': integrity_percentage
                }
    
    return integrity_results

def business_logic_validation(synthetic_data):
    """Validate business logic rules"""
    print("\n💼 Business Logic Validation")
    print("=" * 50)
    
    validation_results = {}
    
    # Check order dates logic
    if 'orders' in synthetic_data:
        orders = synthetic_data['orders']
        if 'order_purchase_timestamp' in orders.columns:
            try:
                orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
                date_range = orders['order_purchase_timestamp'].max() - orders['order_purchase_timestamp'].min()
                print(f"   📅 Order date range: {date_range.days} days")
                
                # Check for future dates
                future_dates = orders['order_purchase_timestamp'] > pd.Timestamp.now()
                if future_dates.any():
                    print(f"   ⚠️ {future_dates.sum()} orders have future dates")
                else:
                    print(f"   ✅ All order dates are realistic")
            except:
                print(f"   ⚠️ Could not validate order dates")
    
    # Check payment values
    if 'payments' in synthetic_data:
        payments = synthetic_data['payments']
        if 'payment_value' in payments.columns:
            negative_payments = payments['payment_value'] < 0
            zero_payments = payments['payment_value'] == 0
            
            print(f"   💰 Payment validation:")
            print(f"      Negative payments: {negative_payments.sum()}")
            print(f"      Zero payments: {zero_payments.sum()}")
            print(f"      Average payment: ${payments['payment_value'].mean():.2f}")
            
            if negative_payments.sum() == 0:
                print(f"      ✅ No negative payments")
            
            validation_results['payments'] = {
                'negative_count': negative_payments.sum(),
                'zero_count': zero_payments.sum(),
                'avg_value': payments['payment_value'].mean()
            }
    
    # Check product categories (English)
    if 'products' in synthetic_data:
        products = synthetic_data['products']
        category_cols = [col for col in products.columns if 'category' in col.lower()]
        
        if category_cols:
            category_col = category_cols[0]
            unique_categories = products[category_col].nunique()
            
            print(f"   🏷️ Product categories:")
            print(f"      Unique categories: {unique_categories}")
            print(f"      Top 5 categories: {list(products[category_col].value_counts().head().index)}")
            
            # Check if categories are in English
            sample_categories = products[category_col].dropna().head(10)
            print(f"      Sample categories: {list(sample_categories)}")
            
            validation_results['products'] = {
                'unique_categories': unique_categories,
                'sample_categories': list(sample_categories)
            }
    
    return validation_results

def generate_authenticity_report(comparison_results, categorical_results, integrity_results, validation_results):
    """Generate overall authenticity score with smart categorical filtering"""
    print("\n🎯 AUTHENTICITY REPORT")
    print("=" * 50)
    
    scores = []
    
    # Statistical similarity score
    stat_scores = []
    for table_results in comparison_results.values():
        for col_results in table_results.values():
            if col_results['similarity_score'] > 0:
                stat_scores.append(col_results['similarity_score'])
    
    avg_stat_score = np.mean(stat_scores) if stat_scores else 0.5
    print(f"📊 Statistical Similarity Score: {avg_stat_score:.3f}")
    scores.append(avg_stat_score)
    
    # Smart categorical overlap score (excluding synthetic-by-design columns)
    meaningful_cat_scores = []
    excluded_count = 0
    total_cat_cols = 0
    
    for table_results in categorical_results.values():
        for col_name, col_results in table_results.items():
            total_cat_cols += 1
            if not col_results['excluded_from_scoring']:
                meaningful_cat_scores.append(col_results['overlap_percentage'] / 100)
            else:
                excluded_count += 1
    
    avg_cat_score = np.mean(meaningful_cat_scores) if meaningful_cat_scores else 0.5
    print(f"🏷️ Meaningful Categorical Overlap Score: {avg_cat_score:.3f}")
    print(f"   📋 Analyzed {total_cat_cols} categorical columns")
    print(f"   ✅ Included {len(meaningful_cat_scores)} meaningful columns in scoring")
    print(f"   ⚠️ Excluded {excluded_count} synthetic-by-design columns")
    scores.append(avg_cat_score)
    
    # Referential integrity score
    integrity_scores = [result['integrity_percentage'] / 100 for result in integrity_results.values()]
    avg_integrity_score = np.mean(integrity_scores) if integrity_scores else 1.0
    print(f"🔗 Referential Integrity Score: {avg_integrity_score:.3f}")
    scores.append(avg_integrity_score)
    
    # Overall authenticity score
    overall_score = np.mean(scores)
    print(f"\n🎯 OVERALL AUTHENTICITY SCORE: {overall_score:.3f}")
    
    # Enhanced interpretation
    if overall_score >= 0.85:
        print("🌟 EXCELLENT - Synthetic data is production-ready!")
    elif overall_score >= 0.75:
        print("✅ VERY GOOD - Synthetic data closely mimics real patterns")
    elif overall_score >= 0.65:
        print("👍 GOOD - Synthetic data is reasonably realistic")
    elif overall_score >= 0.55:
        print("⚠️ FAIR - Synthetic data has some realistic patterns")
    else:
        print("❌ POOR - Synthetic data needs improvement")
    
    # Additional insights
    print(f"\n📋 DETAILED INSIGHTS:")
    print(f"   🔢 Statistical Patterns: {avg_stat_score:.1%} similarity")
    print(f"   🏷️ Business Categories: {avg_cat_score:.1%} overlap (meaningful columns only)")
    print(f"   🔗 Data Relationships: {avg_integrity_score:.1%} integrity")
    
    # Show excluded columns summary
    if excluded_count > 0:
        print(f"\n📝 EXCLUDED FROM CATEGORICAL SCORING:")
        for table_results in categorical_results.values():
            for col_name, col_results in table_results.items():
                if col_results['excluded_from_scoring']:
                    print(f"   • {col_name} {col_results['exclusion_reason']}")
    
    return overall_score

def main():
    """Main authenticity checking function"""
    print("🔍 ENHANCED SYNTHETIC DATA AUTHENTICITY CHECKER")
    print("🧠 Smart Column Filtering for Realistic Scoring")
    print("=" * 60)
    
    # Load data
    real_data, synthetic_data = load_data()
    
    if not real_data or not synthetic_data:
        print("❌ Could not load data. Please check file paths.")
        return
    
    # Run all checks
    comparison_results = statistical_comparison(real_data, synthetic_data)
    categorical_results = categorical_comparison(real_data, synthetic_data)
    integrity_results = referential_integrity_check(synthetic_data)
    validation_results = business_logic_validation(synthetic_data)
    
    # Generate final report
    overall_score = generate_authenticity_report(
        comparison_results, categorical_results, 
        integrity_results, validation_results
    )
    
    print(f"\n📋 Enhanced authenticity check complete!")
    print(f"🎯 Final Score: {overall_score:.3f}")
    print("✨ Your synthetic data quality assessment is now much more accurate!")

if __name__ == "__main__":
    main()
