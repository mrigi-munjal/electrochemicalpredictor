# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Add KNN and Decision Tree to R^2 scores dictionary
r2_scores = {
    "Linear Regression": [],
    "MLP Regressor": [],
    "Random Forest Regressor": [],
    "KNN Regressor": [],
    "Decision Tree Regressor": []
}
target_property_list = ['voltage', 'capacity', 'energy', 'conductivity','columbic_efficiency']

gnn_data = pd.read_csv('../ALIGNN-BERT-TL-crystal/data/embeddings/data0.csv')
gnn_data = gnn_data.drop(columns=['full'])
# %%
for target_property in target_property_list:
    df1 = pd.read_excel(f'./pre-processed/{target_property}.xlsx')
    # Compute average voltage grouped by 'jid'
    avg_df = df1.groupby('jid')['Value'].mean().reset_index()
    avg_df = avg_df[avg_df['jid'].astype(str).str.startswith('JVASP')]

    # Merge with the second dataframe to align features with the average voltage
    merged_df = pd.merge(avg_df, gnn_data, left_on='jid', right_on='id')
    
    merged_df = merged_df[merged_df['jid'] == merged_df['id']]
    
    # Construct X (feature vectors) and y (target values)
    X = merged_df.drop(columns=['Value', 'id', 'jid']).values
    y = merged_df['Value'].values
    
    print(X.shape, y.shape)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    # Train and evaluate Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_test_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_test_pred_lr)

    # Train and evaluate MLP Regressor
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,100,), max_iter=10000, random_state=42)
    mlp_model.fit(X_train, y_train)
    y_test_pred_mlp = mlp_model.predict(X_test)
    r2_mlp = r2_score(y_test, y_test_pred_mlp)

    # Train and evaluate Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_test_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_test_pred_rf)

    # Train and evaluate KNN Regressor
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_test_pred_knn = knn_model.predict(X_test)
    r2_knn = r2_score(y_test, y_test_pred_knn)

    # Train and evaluate Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_test_pred_dt = dt_model.predict(X_test)
    r2_dt = r2_score(y_test, y_test_pred_dt)

    # Print R^2 scores
    print(f"Linear Regression Test R^2: {r2_lr:.4f}")
    print(f"MLP Regressor Test R^2: {r2_mlp:.4f}")
    print(f"Random Forest Regressor Test R^2: {r2_rf:.4f}")
    print(f"KNN Regressor Test R^2: {r2_knn:.4f}")
    print(f"Decision Tree Regressor Test R^2: {r2_dt:.4f}")
    
    r2_scores["Linear Regression"].append(r2_lr)
    r2_scores["MLP Regressor"].append(r2_mlp)
    r2_scores["Random Forest Regressor"].append(r2_rf)
    r2_scores["KNN Regressor"].append(r2_knn)
    r2_scores["Decision Tree Regressor"].append(r2_dt)

# %%
plt.rcParams.update({'font.size': 10})
# Update the plotting section to include KNN and Decision Tree
x = np.arange(len(target_property_list))  # the label locations
width = 0.15  # the width of the bars
fig, ax = plt.subplots(figsize=(6, 4))
rects1 = ax.bar(x - 2*width, r2_scores["Linear Regression"], width, label='Linear Regression')
rects2 = ax.bar(x - width, r2_scores["MLP Regressor"], width, label='MLP Regressor')
rects3 = ax.bar(x, r2_scores["Random Forest Regressor"], width, label='Random Forest Regressor')
rects4 = ax.bar(x + width, r2_scores["KNN Regressor"], width, label='KNN Regressor')
rects5 = ax.bar(x + 2*width, r2_scores["Decision Tree Regressor"], width, label='Decision Tree Regressor')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Target Property')
ax.set_ylabel('Test R^2 Score')
ax.set_title('Test R^2 Scores by Model and Target Property')
ax.set_xticks(x)
ax.set_xticklabels(target_property_list)
ax.legend()
ax.set_ylim(0, 1)

# Add value annotations on top of bars
def add_annotations(rects):
    """Add annotations on top of the bars."""
    for rect in rects:
        height = max(rect.get_height(), 0)
        ax.annotate(f'{rect.get_height():.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_annotations(rects1)
add_annotations(rects2)
add_annotations(rects3)
add_annotations(rects4)
add_annotations(rects5)

fig.tight_layout()
plt.show()
# %%
