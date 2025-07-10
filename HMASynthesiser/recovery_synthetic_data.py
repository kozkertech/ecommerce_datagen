#!/usr/bin/env python3
"""
Recovery Script: Load trained HMASynthesizer and generate synthetic CSV files
This script loads your saved model and generates the synthetic data without visualization issues
"""

import pandas as pd
from sdv.multi_table import HMASynthesizer
import os

def main():
    # Configuration
    MODEL_PATH = 'brazilian_ecommerce_hma_english.pkl'  # Your saved model
    SCALE = 0.1  # Generate 10% of original data size
    
    print("🔄 Recovery Script: Generating synthetic data from saved model")
    print("=" * 60)
    
    # Step 1: Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.pkl'):
                print(f"   📁 {file}")
        return
    
    print(f"✅ Found model file: {MODEL_PATH}")
    
    # Step 2: Load the trained synthesizer
    print("\n📥 Loading trained HMASynthesizer...")
    try:
        synthesizer = HMASynthesizer.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return
    
    # Step 3: Generate synthetic data
    print(f"\n🎯 Generating synthetic data (scale: {SCALE})...")
    try:
        synthetic_data = synthesizer.sample(scale=SCALE)
        print("✅ Synthetic data generation completed!")
    except Exception as e:
        print(f"❌ Failed to generate synthetic data: {str(e)}")
        return
    
    # Step 4: Display summary of generated data
    print("\n📊 Generated synthetic data summary:")
    total_synthetic_rows = 0
    for table_name, df in synthetic_data.items():
        total_synthetic_rows += len(df)
        print(f"   {table_name}: {len(df):,} rows × {df.shape[1]} columns")
        
        # Special check for products with English categories
        if table_name == 'products':
            category_cols = [col for col in df.columns if 'category' in col.lower()]
            if category_cols:
                category_col = category_cols[0]
                unique_categories = df[category_col].nunique()
                print(f"      📈 English categories: {unique_categories} unique")
                top_categories = df[category_col].value_counts().head(3)
                print(f"      🏆 Top categories: {list(top_categories.index)}")
    
    print(f"\n📊 Total synthetic rows generated: {total_synthetic_rows:,}")
    
    # Step 5: Save all synthetic tables as CSV files
    print("\n💾 Saving synthetic data to CSV files...")
    saved_files = []
    
    for table_name, df in synthetic_data.items():
        filename = f'synthetic_{table_name}_hma_english.csv'
        try:
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            print(f"   ✅ Saved: {filename} ({len(df):,} rows)")
        except Exception as e:
            print(f"   ❌ Failed to save {filename}: {str(e)}")
    
    # Step 6: Create summary report
    print("\n📋 Creating summary report...")
    summary_data = []
    for table_name, df in synthetic_data.items():
        summary_data.append({
            'Table': table_name,
            'Rows': len(df),
            'Columns': df.shape[1],
            'File': f'synthetic_{table_name}_hma_english.csv'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('synthetic_data_summary.csv', index=False)
    print("   ✅ Saved: synthetic_data_summary.csv")
    
    # Step 7: Final success message
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Synthetic data recovery completed!")
    print("=" * 60)
    print(f"📁 Generated {len(saved_files)} CSV files:")
    for filename in saved_files:
        print(f"   - {filename}")
    print("\n📊 Summary:")
    print(f"   - Total tables: {len(synthetic_data)}")
    print(f"   - Total synthetic rows: {total_synthetic_rows:,}")
    print(f"   - Model used: {MODEL_PATH}")
    print("\n✅ Your Brazilian e-commerce synthetic data with English categories is ready!")
    
    # Step 8: Quick data validation
    print("\n🔍 Quick validation check:")
    for table_name, df in synthetic_data.items():
        null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        print(f"   {table_name}: {null_percentage:.1f}% null values")
    
    print("\n🎯 Next steps:")
    print("   1. Check the CSV files in your current directory")
    print("   2. Open synthetic_data_summary.csv for an overview") 
    print("   3. Use the synthetic data for your e-commerce analysis!")

if __name__ == "__main__":
    main()
