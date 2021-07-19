# CSV to Wav (Batch)
Converts the data CSV files contained in the dataset CSV file into Wav files.

## Reference
The converted Wav files are saved in the “wavfiles” folder created in the folder specified by output-csv.

The bit length of the converted Wav file is 16 bits. The amplitude of the converted Wav file is the value written in the CSV file multiplied by 32,768 and converted to an integer.

