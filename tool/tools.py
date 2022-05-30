from torchmetrics import StructuralSimilarityIndexMeasure
from torch.utils.data import Subset, ConcatDataset
from torchvision.utils import save_image
import os


def ssim_calculate(y_obs, y_pred):
    """Returns the SSIM index between two image tensors.
       Both tensors must be the same size, and 4D dimensions.
       Closer to 1 score is better.
    Args:
        y_actual (tensor): first set of data (the actual)
        y_pred (tensor): second set of data (the predictions)
    Returns:
        float: SSIM index
    """
    ssim = StructuralSimilarityIndexMeasure
    original_ = y_obs.reshape(1, 1, 128, 128)
    prediction_ = y_pred.reshape(1, 1, 128, 128)
    measure = ssim(original_, prediction_)
    return(measure)


def plot_data(data, ts, rows=2, cols=3, figsize=(20, 8)):
    """Plots data as matplotlib subplots.
    Args:
        data (array): 2D-array of data to plot
        ts (array): array of timesteps: e.g. [0,1,2,3,4]
        rows (int, optional): number of rows required. Defaults to 2.
        cols (int, optional): number of columns required. Defaults to 3.
        figsize (tuple, optional):
        size of figure in inches. Defaults to (20,8).
    """
    return()


def read_data_for_forecasts(step, num_simulations):
    """This is a function to read in wildfire
       simulation data from training and test dataset
    Args:
        step (string): What dataset the user wants to read in, test or train?
        num_simulations (int): How many simulations from
        each data set does the user want to include in its forecasting model
    Raises:
        Exception: if the user inputs a nuber of simulations that
        is greater than what is available from the test or train data sets
    Returns:
        array, array : two arrays in the correct format for
        the input and target of the forecasting model
    """

    return


def reshape_data_for_forecast(x):
    """Function to reshape array into the correct
        dimensions for the input and target of forecasting model
    Args:
        x (array): Array that the user wants to reshape
    Raises:
        Exception: if inputted array is not 4
        dimensional it cannot be reshape to correct format
        (needs to be in format (num_simulations, 3, 871, 913))
    Returns:
        array: array in correct format for forecasting model
        (num_simulation, 1, 871*913)
    """
    return


def Insert_row(row_number, df, row_value):
    """
    Credit for this function belongs to GeeksforGeeks

    Function to insert a row into a dataframe at position row_number

    Args:
        row_number (int): Position to place the new row
        df (dataframe): dataframe to place the new row into
        row_value (list): new row to place in the dataframe
    Returns:
        new dataframe with the desired row added
    """
    # Starting value of upper half
    start_upper = 0
    # End value of upper half
    end_upper = row_number
    # Start value of lower half
    start_lower = row_number
    # End value of lower half
    end_lower = df.shape[0]
    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]
    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]
    # Combine the two lists
    index_ = upper_half + lower_half
    # Update the index of the dataframe
    df.index = index_
    # Insert a row at the end
    df.loc[row_number] = row_value
    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df


def insert_point(index, dataset, new_point):
    """
    Function to insert a datapoint in a dataset the
    datapoint becomes dataset[index],
    the datapoint previously at dataset[index] becomes dataset[index+1]

    Args:
        index (int): Index to place the new datapoint at
        dataset (tuple): Dataset to insert the new datapoint in
        new_point (tuple): New datapoint to insert in the dataset
    Returns:
        dataset (tuple): Dataset with new_point inserted into
        the into it at position index.


    """
    set_a = Subset(dataset, list(range(0, index)))
    set_b = Subset(dataset, list(range(index, len(dataset))))
    new_set = ConcatDataset([set_a, [new_point], set_b])
    return new_set


def standardise(df, dataset):
    """
    Function to standardise the timings for a dataset.

    Args:
        df (dataframe): Array to standardise, this must contain data
        for only 1 storm.
        dataset (tuple): The corresponding dataset
        of images for the dataframe df.
    Raises:
        Exception: if the dataset contains data for more than one storm
        Exception: if the dataset and dataframe do not correspond
    Returns:
        df (dataframe): dataframe with standardised timings that are
        approximately uniform
        dataset (tuple): a dataset that corresponds
        to the dataframe to be returned
    """
    # The extension for the storm.
    ext = df['Image ID'].iloc[0][0:3]
    # iterates through the dataframe, and the dataset of images
    N = len(df)
    N_dataset = len(dataset)
    i = 0
    if N != N_dataset:
        raise Exception("The dataset must correspond exactly to the dataframe")
    while i < N - 1:
        if ext not in df['Image ID'].iloc[i] or \
               ext not in df['Storm ID'].iloc[i]:
            raise Exception("The arguments must\
             contain data only for 1 storm.")
        if int(float(df['Relative Time'].iloc[i + 1])) - int(float(df[
            'Relative Time'].iloc[i])) > 1790 and (int(float(df[
                'Relative Time'].iloc[i + 1])) - int(float(df[
                    'Relative Time'].iloc[i]))) < 1810:
            # no need to insert datapoint here, the interval is approx 1800
            pass
        elif int(float(df['Relative Time'].iloc[i + 1])) - int(float(df[
            'Relative Time'].iloc[i])) > 3590 and int(float(df[
                'Relative Time'].iloc[i + 1])) - int(float(df[
                    'Relative Time'].iloc[i])) < 3610:
            # insert datapoint if interval is approx 3600
            row_value = [f'{ext}_fake_{i}point5', ext,
                         str((int(float(df['Relative Time'].iloc[
                             i + 1])) + int(float(df[
                                 'Relative Time'].iloc[i])))//2),
                         df['Ocean'].iloc[i], str((int(float(df[
                             'Wind Speed'].iloc[
                          i + 1])) + int(float(df['Wind Speed'].iloc[i])))//2)]
            df = Insert_row(i+1, df, row_value)
            new_point = (dataset[i+1][0]/2 + dataset[i][0]/2,
                         dataset[i+1][1]/2 + dataset[i][1]/2)
            dataset = insert_point(index=i+1, dataset=dataset,
                                   new_point=new_point)
            N += 1
        i += 1

    return df, dataset


def add_links(df, dataset):
    """
    Function add links for images to the dataframe in a column 'Links'

    Args:
        df (dataframe): Array to add the column 'Links' to.
        dataset (tuple): The corresponding dataset for the dataframe df
    Raises:
        Exception: if the dataset and dataframe do not correspond
    Returns:
        dataframe: dataframe with the added column 'Links'
    """
    N = len(df)
    N_dataset = len(dataset)
    i = 0
    if N != N_dataset:
        raise Exception("The dataset must correspond exactly to the dataframe")

    df_links = []
    str1 = 'data/nasa_tropical_storm_competition_train_source'
    str2 = '/nasa_tropical_storm_competition_train_source_'
    for i in range(len(df)):
        ext = df['Image ID'].iloc[i]
        if 'fake' in ext:
            directory = 'nasa_tropical_storm_competition_train_source_' + ext
            parent_dir = 'data/nasa_tropical_storm_competition_train_source'
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            save_image(dataset[i][0][0], str1 + str2 + ext + '/image.jpg')
        df_links += [str1 + str2 + ext + '/image.jpg']

    df['Links'] = df_links
    return df
