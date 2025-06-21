import pandas as pd

def convert_txt_to_csv(input_txt):

    
    column_names = ['unit_number', 'time', 'op1', 'op2', 'op3'] + [f'{i}' for i in range(1, 22)]

    # read the test data file into a DataFrame
    df = pd.read_csv(f'{input_txt}.txt', sep='\\s+', header=None, names=column_names)

    print(df.shape)

    # save to CSV file
    df.to_csv(f'{input_txt}.csv', index=False)


# Run the conversion
convert_txt_to_csv("train_FD002")
