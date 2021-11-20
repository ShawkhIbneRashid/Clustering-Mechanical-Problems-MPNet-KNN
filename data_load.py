from dataset import Load_Preprocess

directory = '../Dataset_Clusterin_Problems.csv'
load_preprocess = Load_Preprocess(directory)
df = load_preprocess.load_data()
print("Showing the first 20 rows in the dataset before applying preprocessing techniques")
#showing the first 20 rows of the dataset before applying preprocessing techniques
print(df[0:20])
#showing the first 20 rows of the dataset after applying preprocessing techniques
df = load_preprocess.preprocess(df)
print("Showing the first 20 rows in the dataset after applying preprocessing techniques")
print(df[0:20])
