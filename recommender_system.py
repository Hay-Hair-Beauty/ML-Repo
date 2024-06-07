import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset (assuming it's saved as a CSV named "data_recomm.csv")
haircare = pd.read_csv('data_recomm.csv', low_memory=False)

# Data Cleaning: Remove duplicates (optional)
haircare.drop_duplicates(inplace=True)


# Function to split and create dummy variables for hair issues
def create_hair_issue_columns(data):
    # Split hair issues into a list
    hair_issue = data['hair_issue'].str.split(', ').tolist()
    # Flatten the list
    flat_hair_issue = [item for sublist in hair_issue for item in sublist]
    # Get unique hair issues
    unique_hair_issue = list(set(flat_hair_issue))

    # Add columns for each unique hair issue
    for issue in unique_hair_issue:
        data[issue] = 0

    # Update data with dummy values
    for index, row in data.iterrows():
        for val in row.hair_issue.split(', '):
            if val != 'NA':
                data.loc[index, val] = 1
    return data


# Clean data and create dummy variables
haircare = create_hair_issue_columns(haircare.copy())

# Modeling with Content Based Filtering
# Initializing TfidfVectorizer
tf = TfidfVectorizer()

# Perform IDF calculation on 'hair_issue' data
tf.fit(haircare['hair_issue'])

# Transforming data to matrix form
tfidf_matrix = tf.transform(haircare['hair_issue'])

# Calculating Cosine Similarity on the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Creating a dataframe from the cosine_similarity variable
cosine_sim_df = pd.DataFrame(cosine_sim, index=haircare['product_name'], columns=haircare['product_name'])


# Function to get recommendations based on hair_issue
def haircare_recommendations_by_issue(hair_issue, similarity_data=cosine_sim_df,
                                      items=haircare[['product_name', 'hair_issue']], k=5):
    # Get products targeting the hair issue
    matching_products = items[items['hair_issue'].str.contains(hair_issue)]

    if matching_products.empty:
        return pd.DataFrame()  # Return empty DataFrame if no matching products found

    # Select the first matching product
    selected_product = matching_products.iloc[0]['product_name']

    # Retrieve similarity scores for the chosen product
    similarity_array = similarity_data.loc[:, selected_product].to_numpy().squeeze()
    # Get indexes of the most similar products (descending order)
    index = np.argsort(similarity_array)[-k - 1:-1][::-1]

    # Retrieve product names based on the indexes
    closest = similarity_data.columns.to_numpy()[index]

    # Create DataFrame with recommendations and merge with hair_issue data
    closest_df = pd.DataFrame(closest, columns=['product_name'])
    return closest_df.merge(items, on='product_name').head(k)


# Example usage: Get recommendations for Alopecia Areata
print(haircare_recommendations_by_issue('Alopecia Areata'))
