# Data Preprocessing and Encoding for UCI ML Dataset

## Overview

This project involves preprocessing and encoding a dataset obtained from the UCI Machine Learning Repository. The dataset includes various features related to dietary habits, demographics, and lifestyle.

## Dataset

The dataset includes the following columns:
- `GPA`
- `Gender`
- `breakfast`
- `calories_chicken`
- `calories_day`
- `calories_scone`
- `coffee`
- `comfort_food`
- `comfort_food_reasons`
- `comfort_food_reasons_coded`
- `cook`
- `cuisine`
- `diet_current`
- `diet_current_coded`
- `drink`
- `eating_changes`
- `eating_changes_coded`
- `eating_changes_coded1`
- `eating_out`
- `employment`
- `ethnic_food`
- `exercise`
- `father_education`
- `father_profession`
- `fav_cuisine`
- `fav_cuisine_coded`
- `fav_food`
- `food_childhood`
- `fries`
- `fruit_day`
- `grade_level`
- `greek_food`
- `healthy_feeling`
- `healthy_meal`
- `ideal_diet`
- `ideal_diet_coded`
- `income`
- `indian_food`
- `italian_food`
- `life_rewarding`
- `marital_status`
- `meals_dinner_friend`
- `mother_education`
- `mother_profession`
- `nutritional_check`
- `on_off_campus`
- `parents_cook`
- `pay_meal_out`
- `persian_food`
- `self_perception_weight`
- `soup`
- `sports`
- `thai_food`
- `tortilla_calories`
- `turkey_calories`
- `type_sports`
- `veggies_day`
- `vitamins`
- `waffle_calories`
- `weight`

## Preprocessing Steps

1. **Load the Dataset**
   ```python
   import pandas as pd
   df = pd.read_csv('path/to/your/dataset.csv')
   ```

2. **Explore the Dataset**
   ```python
   print(df.head())
   print(df.info())
   print(df.describe())
   ```

3. **Handle Missing Values**
   ```python
   df.fillna(df.median(), inplace=True)
   df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)
   ```

4. **Identify Categorical Columns**
   ```python
   categorical_columns = df.select_dtypes(include=['object']).columns
   print(categorical_columns)
   ```

5. **Label Encoding**
   ```python
   from sklearn.preprocessing import LabelEncoder
   label_encoder = LabelEncoder()
   df['Gender'] = label_encoder.fit_transform(df['Gender'])
   ```

6. **One-Hot Encoding**
   ```python
   df = pd.get_dummies(df, columns=['cuisine'], drop_first=True)
   ```

7. **Ordinal Encoding**
   ```python
   from sklearn.preprocessing import OrdinalEncoder
   ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
   df['healthy_feeling'] = ordinal_encoder.fit_transform(df[['healthy_feeling']])
   ```

8. **Feature Scaling**
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   df[['calories_chicken', 'calories_day', 'calories_scone']] = scaler.fit_transform(df[['calories_chicken', 'calories_day', 'calories_scone']])
   ```

9. **Handle Redundant or Irrelevant Columns**
   ```python
   df.drop(columns=['comfort_food_reasons_coded'], inplace=True)
   ```

10. **Split the Dataset**
    ```python
    from sklearn.model_selection import train_test_split
    X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual target
    y = df['target_column']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

11. **Save Preprocessed Data (Optional)**
    ```python
    df.to_csv('preprocessed_food_coded.csv', index=False)
    ```

## Requirements

- `pandas`
- `scikit-learn`

Install the required packages using:

```bash
pip install pandas scikit-learn
```

## Usage

1. Load the dataset using the provided code.
2. Apply preprocessing steps as described.
3. Split the dataset and save the processed data for further use.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
