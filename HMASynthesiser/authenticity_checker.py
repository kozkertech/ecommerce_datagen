#!/usr/bin/env python3


import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
REAL_DATA_PATH = r'E:\brazilian dataset'
SYNTH_DIR = '.'  # directory where synthetic CSVs live
TRANSLATION_CSV = os.path.join(REAL_DATA_PATH, 'product_category_name_translation.csv')

EXCLUDE_FROM_CATEGORICAL_SCORING = {
    'order_id', 'customer_id', 'product_id', 'seller_id', 'review_id',
    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
    'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date',
    'review_creation_date', 'review_answer_timestamp', 'customer_unique_id',
    'customer_city', 'customer_state', 'seller_city', 'seller_state', 'review_comment_message', 'seller_zip_code_prefix',
    'customer_zip_code_prefix', 'geolocation_zip_code_prefix'
}


print("🔄 Loading translation map...")
try:
    trans_df = pd.read_csv(TRANSLATION_CSV)
    translation_map = dict(zip(
        trans_df['product_category_name'],
        trans_df['product_category_name_english']
    ))
    print(f"✅ Translation map loaded successfully: {len(translation_map)} mappings")
    print(f"📋 Sample translations:")
    for i, (port, eng) in enumerate(list(translation_map.items())[:3]):
        print(f"   {port} → {eng}")
except Exception as e:
    print(f"❌ Could not load translation map: {str(e)}")
    translation_map = {}


def load_data():
    """Load real and synthetic tables into dictionaries"""
    print("\n📥 Loading datasets...")
    print("=" * 50)
    
    real_files = {
        'orders':'olist_orders_dataset.csv',
        'order_items':'olist_order_items_dataset.csv',
        'products':'olist_products_dataset.csv',
        'payments':'olist_order_payments_dataset.csv',
        'customers':'olist_customers_dataset.csv',
        'sellers':'olist_sellers_dataset.csv',
        'reviews':'olist_order_reviews_dataset.csv'
    }
    
    real_data, synthetic_data = {}, {}
    
    # Load real data
    print("📊 Loading Real Data:")
    for name, fn in real_files.items():
        try:
            real_data[name] = pd.read_csv(os.path.join(REAL_DATA_PATH, fn))
            print(f"   ✅ Loaded real {name}: {real_data[name].shape}")
        except Exception as e:
            print(f"   ❌ Could not load real {name}: {str(e)}")
            real_data[name] = pd.DataFrame()
    
    # Load synthetic data  
    print("\n🤖 Loading Synthetic Data:")
    for name in real_files.keys():
        try:
            synthetic_data[name] = pd.read_csv(os.path.join(SYNTH_DIR, f"synthetic_{name}_hma_english.csv"))
            print(f"   ✅ Loaded synthetic {name}: {synthetic_data[name].shape}")
        except Exception as e:
            print(f"   ❌ Could not load synthetic {name}: {str(e)}")
            synthetic_data[name] = pd.DataFrame()
    
    return real_data, synthetic_data


def statistical_comparison(real_data, synthetic_data):
    """Compare numeric columns via mean/std and KS-test"""
    print("\n📊 Statistical Comparison Analysis")
    print("=" * 50)
    
    comparison_results = {}
    
    for table in real_data:
        rd, sd = real_data[table], synthetic_data.get(table, pd.DataFrame())
        if rd.empty or sd.empty: 
            print(f"\n⚠️ Skipping {table} - missing data")
            continue
            
        print(f"\n🔍 {table.upper()} Analysis:")
        
        numerical_cols = rd.select_dtypes(include=[np.number]).columns.intersection(sd.columns)
        table_results = {}
        
        if len(numerical_cols) == 0:
            print("   📋 No numerical columns found")
            continue
            
        for col in numerical_cols:
            r, s = rd[col].dropna(), sd[col].dropna()
            
            if len(r) == 0 or len(s) == 0:
                print(f"   ⚠️ {col}: No data available")
                continue
                
            # Calculate statistics
            real_mean, synth_mean = r.mean(), s.mean()
            real_std, synth_std = r.std(), s.std()
            real_median, synth_median = r.median(), s.median()
            real_min, synth_min = r.min(), s.min()
            real_max, synth_max = r.max(), s.max()
            
            # Calculate differences
            mean_diff = abs(real_mean - synth_mean) / real_mean * 100 if real_mean != 0 else 0
            std_diff = abs(real_std - synth_std) / real_std * 100 if real_std != 0 else 0
            
            # KS test for distribution similarity
            try:
                ks_stat, p_value = stats.ks_2samp(r, s)
                similarity_score = 1 - ks_stat
            except Exception:
                similarity_score = np.nan
                
            print(f"   📈 {col}:")
            print(f"      Mean: Real={real_mean:.2f}, Synthetic={synth_mean:.2f} ({mean_diff:.1f}% diff)")
            print(f"      Std:  Real={real_std:.2f}, Synthetic={synth_std:.2f} ({std_diff:.1f}% diff)")
            print(f"      Median: Real={real_median:.2f}, Synthetic={synth_median:.2f}")
            print(f"      Range: Real=[{real_min:.2f}, {real_max:.2f}], Synthetic=[{synth_min:.2f}, {synth_max:.2f}]")
            
          
            table_results[col] = {
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'similarity_score': similarity_score if not np.isnan(similarity_score) else 0.5
            }
            
        comparison_results[table] = table_results
        
    return comparison_results


def categorical_comparison(real_data, synthetic_data):
    """Compare categorical overlap, translating product_category_name to English"""
    print("\n🏷️ Categorical Distribution Analysis")
    print("=" * 50)
    
    categorical_results = {}
    
    for table in real_data:
        rd, sd = real_data[table], synthetic_data.get(table, pd.DataFrame())
        if rd.empty or sd.empty: 
            print(f"\n⚠️ Skipping {table} - missing data")
            continue
            
        print(f"\n🔍 {table.upper()} Categorical Analysis:")
        
        categorical_cols = rd.select_dtypes(include=['object']).columns.intersection(sd.columns)
        table_results = {}
        
        if len(categorical_cols) == 0:
            print("   📋 No categorical columns found")
            continue
            
        for col in categorical_cols:
            # Determine exclusion status
            excluded = col in EXCLUDE_FROM_CATEGORICAL_SCORING
            exclusion_reason = ""
            
            if excluded:
                if any(id_term in col.lower() for id_term in ['id', 'unique']):
                    exclusion_reason = "(ID - expected 0%)"
                elif any(date_term in col.lower() for date_term in ['date', 'timestamp']):
                    exclusion_reason = "(Date - expected 0%)"
                elif any(geo_term in col.lower() for geo_term in ['city', 'state', 'zip']):
                    exclusion_reason = "(Geographic - may be synthetic)"
                else:
                    exclusion_reason = "(Synthetic by design)"
            
            # Apply translation for product categories
            english_comparison_note = ""
            if table == 'products' and col == 'product_category_name':
                print(f"   🌐 Applying English translation for {col}...")
                real_vals = rd[col].map(lambda x: translation_map.get(x, x))
                english_comparison_note = "🌐 (English vs English)"
                
                # Show translation sample
                original_sample = rd[col].dropna().head(3).tolist()
                translated_sample = real_vals.dropna().head(3).tolist()
                print(f"      Translation sample:")
                for orig, trans in zip(original_sample, translated_sample):
                    if orig != trans:
                        print(f"        {orig} → {trans}")
                
            else:
                real_vals = rd[col]
            
            synth_vals = sd[col]
            
            # Calculate overlap
            real_set = set(real_vals.dropna().unique())
            synth_set = set(synth_vals.dropna().unique())
            overlap = len(real_set & synth_set)
            total_real = len(real_set)
            pct = overlap / total_real * 100 if total_real > 0 else 0
            
            print(f"   📊 {col} {exclusion_reason} {english_comparison_note}:")
            print(f"      Real categories: {total_real}")
            print(f"      Synthetic categories: {len(synth_set)}")
            print(f"      Category overlap: {overlap}/{total_real} ({pct:.1f}%)")
            
            # Show top categories
            if len(real_set) > 0 and len(synth_set) > 0:
                real_counts = real_vals.value_counts()
                synth_counts = synth_vals.value_counts()
                
                top_real = real_counts.head(3).index.tolist()
                top_synth = synth_counts.head(3).index.tolist()
                
                print(f"      Top real categories: {top_real}")
                print(f"      Top synthetic categories: {top_synth}")
                
                # Check if top categories match
                top_overlap = len(set(top_real[:3]) & set(top_synth[:3]))
                print(f"      Top-3 category overlap: {top_overlap}/3")
                
            # Exclusion status
            if excluded:
                print(f"      ⚠️ Excluded from final categorical score {exclusion_reason}")
            else:
                print(f"      ✅ Included in final categorical score")
                if english_comparison_note:
                    print(f"      🌐 Fair English-to-English comparison applied")
                    
            table_results[col] = {
                'overlap_percentage': pct,
                'real_categories': total_real,
                'synthetic_categories': len(synth_set),
                'excluded_from_scoring': excluded,
                'exclusion_reason': exclusion_reason,
                'english_comparison': bool(english_comparison_note)
            }
            
        categorical_results[table] = table_results
        
    return categorical_results


def referential_integrity_check(synthetic_data):
    """Ensure child tables reference parent keys"""
    print("\n🔗 Referential Integrity Check")
    print("=" * 50)
    
    integrity_results = {}
    
    relationships = [
        ('orders', 'order_items', 'order_id'),
        ('orders', 'payments', 'order_id'),
        ('customers', 'orders', 'customer_id'),
        ('products', 'order_items', 'product_id'),
        ('sellers', 'order_items', 'seller_id')
    ]
    
    for parent_table, child_table, key_col in relationships:
        parent_df = synthetic_data.get(parent_table)
        child_df = synthetic_data.get(child_table)
        
        if parent_df is not None and child_df is not None and not parent_df.empty and not child_df.empty:
            if key_col in parent_df.columns and key_col in child_df.columns:
                parent_keys = set(parent_df[key_col].dropna())
                child_keys = set(child_df[key_col].dropna())
                orphans = child_keys - parent_keys
                
                total_child_records = len(child_df)
                orphan_count = len(orphans)
                integrity_pct = (total_child_records - orphan_count) / total_child_records * 100
                
                print(f"   🔍 {parent_table} → {child_table} ({key_col}):")
                print(f"      Parent keys: {len(parent_keys):,}")
                print(f"      Child records: {total_child_records:,}")
                print(f"      Orphaned records: {orphan_count:,}")
                print(f"      Integrity: {integrity_pct:.1f}%")
                
                if orphan_count == 0:
                    print(f"      ✅ Perfect referential integrity!")
                elif integrity_pct >= 95:
                    print(f"      👍 Excellent integrity")
                elif integrity_pct >= 85:
                    print(f"      ⚠️ Good integrity with minor issues")
                else:
                    print(f"      ❌ Poor integrity - many orphaned records")
                    
                integrity_results[f"{parent_table}_{child_table}"] = {
                    'orphaned_count': orphan_count,
                    'integrity_percentage': integrity_pct
                }
            else:
                print(f"   ⚠️ {parent_table} → {child_table}: Missing {key_col} column")
        else:
            print(f"   ⚠️ {parent_table} → {child_table}: Missing table data")
            
    return integrity_results


def business_logic_validation(synthetic_data):
    """Validate business logic rules"""
    print("\n💼 Business Logic Validation")
    print("=" * 50)
    
    validation_results = {}
    
    # Check order dates logic
    if 'orders' in synthetic_data and not synthetic_data['orders'].empty:
        orders = synthetic_data['orders']
        print("   📅 Order Date Validation:")
        
        if 'order_purchase_timestamp' in orders.columns:
            try:
                orders_copy = orders.copy()
                orders_copy['order_purchase_timestamp'] = pd.to_datetime(orders_copy['order_purchase_timestamp'])
                
                date_range = orders_copy['order_purchase_timestamp'].max() - orders_copy['order_purchase_timestamp'].min()
                print(f"      Date range: {date_range.days} days")
                print(f"      Earliest order: {orders_copy['order_purchase_timestamp'].min()}")
                print(f"      Latest order: {orders_copy['order_purchase_timestamp'].max()}")
                
                # Check for future dates
                future_dates = orders_copy['order_purchase_timestamp'] > pd.Timestamp.now()
                future_count = future_dates.sum()
                
                if future_count == 0:
                    print(f"      ✅ All {len(orders_copy):,} order dates are realistic")
                else:
                    print(f"      ⚠️ {future_count:,} orders have future dates")
                    
            except Exception as e:
                print(f"      ❌ Could not validate order dates: {str(e)}")
        else:
            print(f"      ⚠️ No order_purchase_timestamp column found")
    
    # Check payment values
    if 'payments' in synthetic_data and not synthetic_data['payments'].empty:
        payments = synthetic_data['payments']
        print("\n   💰 Payment Value Validation:")
        
        if 'payment_value' in payments.columns:
            negative_payments = (payments['payment_value'] < 0).sum()
            zero_payments = (payments['payment_value'] == 0).sum()
            total_payments = len(payments)
            avg_payment = payments['payment_value'].mean()
            median_payment = payments['payment_value'].median()
            max_payment = payments['payment_value'].max()
            
            print(f"      Total payments: {total_payments:,}")
            print(f"      Negative payments: {negative_payments:,} ({negative_payments/total_payments*100:.1f}%)")
            print(f"      Zero payments: {zero_payments:,} ({zero_payments/total_payments*100:.1f}%)")
            print(f"      Average payment: ${avg_payment:.2f}")
            print(f"      Median payment: ${median_payment:.2f}")
            print(f"      Maximum payment: ${max_payment:.2f}")
            
            if negative_payments == 0:
                print(f"      ✅ No negative payments found")
            else:
                print(f"      ⚠️ Found {negative_payments} negative payment values")
                
            validation_results['payments'] = {
                'negative_count': negative_payments,
                'zero_count': zero_payments,
                'avg_value': avg_payment
            }
        else:
            print(f"      ⚠️ No payment_value column found")
    
    # Check product categories
    if 'products' in synthetic_data and not synthetic_data['products'].empty:
        products = synthetic_data['products']
        print("\n   🏷️ Product Category Validation:")
        
        category_cols = [col for col in products.columns if 'category' in col.lower()]
        
        if category_cols:
            category_col = category_cols[0]
            unique_categories = products[category_col].nunique()
            total_products = len(products)
            
            print(f"      Total products: {total_products:,}")
            print(f"      Unique categories: {unique_categories}")
            
            # Show top categories
            top_categories = products[category_col].value_counts().head(5)
            print(f"      Top 5 categories:")
            for cat, count in top_categories.items():
                pct = count / total_products * 100
                print(f"        {cat}: {count:,} ({pct:.1f}%)")
            
            # Check if categories appear to be in English
            sample_categories = products[category_col].dropna().head(10).tolist()
            english_indicators = ['_', 'bed', 'bath', 'table', 'sports', 'leisure', 'furniture', 'health', 'beauty']
            has_english = any(any(indicator in str(cat).lower() for indicator in english_indicators) for cat in sample_categories)
            
            if has_english:
                print(f"      ✅ Categories appear to be in English format")
            else:
                print(f"      ⚠️ Categories may still be in Portuguese")
                
            validation_results['products'] = {
                'unique_categories': unique_categories,
                'sample_categories': sample_categories[:5],
                'appears_english': has_english
            }
        else:
            print(f"      ⚠️ No category columns found")
    
    return validation_results


def generate_authenticity_report(comparison_results, categorical_results, integrity_results, validation_results):
    """Generate overall authenticity score with smart categorical filtering"""
    print("\n🎯 AUTHENTICITY REPORT")
    print("=" * 50)
    
    scores = []
    
    # Statistical similarity score
    print("📊 Statistical Analysis Results:")
    stat_scores = []
    for table_name, table_results in comparison_results.items():
        print(f"   {table_name}:")
        for col_name, col_results in table_results.items():
            sim_score = col_results['similarity_score']
            mean_diff = col_results['mean_diff']
            if sim_score > 0:
                stat_scores.append(sim_score)
                print(f"     {col_name}: similarity={sim_score:.3f}, mean_diff={mean_diff:.1f}%")
    
    avg_stat_score = np.mean(stat_scores) if stat_scores else 0.5
    print(f"\n📈 Overall Statistical Similarity Score: {avg_stat_score:.3f}")
    scores.append(avg_stat_score)
    
    # Smart categorical overlap score (excluding synthetic-by-design columns)
    print(f"\n🏷️ Categorical Analysis Results:")
    meaningful_cat_scores = []
    excluded_count = 0
    total_cat_cols = 0
    
    for table_name, table_results in categorical_results.items():
        print(f"   {table_name}:")
        for col_name, col_results in table_results.items():
            total_cat_cols += 1
            overlap_pct = col_results['overlap_percentage']
            excluded = col_results['excluded_from_scoring']
            
            if not excluded:
                meaningful_cat_scores.append(overlap_pct / 100)
                status = "✅ Included"
            else:
                excluded_count += 1
                status = f"⚠️ Excluded {col_results['exclusion_reason']}"
            
            print(f"     {col_name}: {overlap_pct:.1f}% {status}")
    
    avg_cat_score = np.mean(meaningful_cat_scores) if meaningful_cat_scores else 0.5
    print(f"\n🏷️ Meaningful Categorical Overlap Score: {avg_cat_score:.3f}")
    print(f"   📋 Analyzed {total_cat_cols} categorical columns")
    print(f"   ✅ Included {len(meaningful_cat_scores)} meaningful columns in scoring")
    print(f"   ⚠️ Excluded {excluded_count} synthetic-by-design columns")
    scores.append(avg_cat_score)
    
    # Referential integrity score
    print(f"\n🔗 Referential Integrity Results:")
    integrity_scores = []
    for rel_name, result in integrity_results.items():
        integrity_pct = result['integrity_percentage']
        integrity_scores.append(integrity_pct / 100)
        print(f"   {rel_name}: {integrity_pct:.1f}%")
    
    avg_integrity_score = np.mean(integrity_scores) if integrity_scores else 1.0
    print(f"\n🔗 Overall Referential Integrity Score: {avg_integrity_score:.3f}")
    scores.append(avg_integrity_score)
    
    # Overall authenticity score
    overall_score = np.mean(scores)
    print(f"\n🎯 OVERALL AUTHENTICITY SCORE: {overall_score:.3f}")
    
    # Enhanced interpretation with detailed breakdown
    print(f"\n📋 SCORE INTERPRETATION:")
    if overall_score >= 0.85:
        print("🌟 EXCELLENT - Synthetic data is production-ready!")
        print("   Your synthetic data demonstrates high fidelity across all dimensions.")
    elif overall_score >= 0.75:
        print("✅ VERY GOOD - Synthetic data closely mimics real patterns")
        print("   Minor improvements may enhance realism further.")
    elif overall_score >= 0.65:
        print("👍 GOOD - Synthetic data is reasonably realistic")  
        print("   Some areas may need attention for production use.")
    elif overall_score >= 0.55:
        print("⚠️ FAIR - Synthetic data has some realistic patterns")
        print("   Significant improvements needed for production use.")
    else:
        print("❌ POOR - Synthetic data needs substantial improvement")
        print("   Review generation parameters and methodology.")
    
    # Detailed component breakdown
    print(f"\n📊 COMPONENT BREAKDOWN:")
    print(f"   🔢 Statistical Patterns: {avg_stat_score:.1%} similarity")
    print(f"   🏷️ Business Categories: {avg_cat_score:.1%} overlap (meaningful columns only)")
    print(f"   🔗 Data Relationships: {avg_integrity_score:.1%} integrity")
    
    # Show excluded columns summary
    if excluded_count > 0:
        print(f"\n📝 COLUMNS EXCLUDED FROM CATEGORICAL SCORING:")
        for table_results in categorical_results.values():
            for col_name, col_results in table_results.items():
                if col_results['excluded_from_scoring']:
                    print(f"   • {col_name} {col_results['exclusion_reason']}")
        print(f"   💡 These columns are expected to have low overlap by design")
    
    return overall_score


def main():
    """Main authenticity checking function"""
    print("🔍 ENHANCED SYNTHETIC DATA AUTHENTICITY CHECKER")
    print("🧠 Smart Column Filtering + English Category Translation")
    print("=" * 60)
    
    # Load data 
    real_data, synthetic_data = load_data()
    
    if not real_data or not synthetic_data:
        print("❌ Could not load data. Please check file paths.")
        return
    
    # Verify data loaded successfully
    real_tables = sum(1 for df in real_data.values() if not df.empty)
    synth_tables = sum(1 for df in synthetic_data.values() if not df.empty)
    
    print(f"\n📋 Data Loading Summary:")
    print(f"   Real tables loaded: {real_tables}/{len(real_data)}")
    print(f"   Synthetic tables loaded: {synth_tables}/{len(synthetic_data)}")
    
    if real_tables == 0 or synth_tables == 0:
        print("❌ Insufficient data loaded. Please check file paths and formats.")
        return
    
    # Run all checks
    print(f"\n🚀 Starting comprehensive authenticity analysis...")
    comparison_results = statistical_comparison(real_data, synthetic_data)
    categorical_results = categorical_comparison(real_data, synthetic_data)
    integrity_results = referential_integrity_check(synthetic_data)
    validation_results = business_logic_validation(synthetic_data)
    
    # Generate final report
    overall_score = generate_authenticity_report(
        comparison_results, categorical_results, 
        integrity_results, validation_results
    )
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"🎯 Final Authenticity Score: {overall_score:.3f}")
    print("✨ Enhanced analysis with fair English-to-English comparison!")
    print("📊 Smart filtering applied to focus on meaningful business patterns")


if __name__ == "__main__":
    main()
