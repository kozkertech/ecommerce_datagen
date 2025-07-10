**SYNTHETIC DATA GENERATOR VIA HMA SYNTHESISER**

**Things to keep in mind before testing the code in your system**
1) Download brazilian dataset from kaggle. Extract the CSV files and store it in a folder.
2) Install the relevant libraries
   ```pip install sdv pandas mathplotlib numpy seaborn``` 
3) Update the DATA_FOLDER_PATH in line 705 to the file path of the brazilian dataset stored in your system.


**recovery_synthetic_data.py**

This code is to load your already trained HMASynthesizer model and generate synthetic CSV files without visualization issues.
This is used if the sythetic csv files need to be generated, as often after model training, the visualisation aspect takes time and we manually end the process.Therefore run this code to generate the files if you wish to see them and test it out.
