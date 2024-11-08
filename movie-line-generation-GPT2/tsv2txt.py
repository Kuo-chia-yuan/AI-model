import pandas as pd

# 打開檔案逐行讀取，並將最後一欄儲存到新的文件中
with open("movie_lines.tsv", "r", encoding="utf-8") as infile, open("training_data.txt", "w", encoding="utf-8") as outfile:
    for line in infile:
        columns = line.strip().split("\t")  # 以制表符分割
        if columns:  # 確保行不為空
            last_column = columns[-1]  # 取得最後一欄
            outfile.write(last_column + "\n")  # 將最後一欄寫入新的文件
