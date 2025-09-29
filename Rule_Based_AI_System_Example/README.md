# Sample submission for the Building a Rule-Based AI System in Python project.

---

## Part 1: Initial Project Ideas

### 1. Project Idea 1: Recipe Recommender
- **Description:** A system that recommends recipes based on ingredients the user has on hand. The user enters ingredients, and the system matches them to recipes using predefined rules.  
- **Rule-Based Approach:**  
  - The system checks for exact matches and partial matches with the ingredients required for recipes in a dataset.  
  - Missing ingredients are suggested for partial matches.

### 2. Project Idea 2: Simple Chatbot
- **Description:** A chatbot that responds to user inputs with predefined answers. The chatbot simulates a conversation by identifying keywords and phrases in user inputs.  
- **Rule-Based Approach:**  
  - Responses are based on keywords such as "hello," "help," or "bye."  
  - For example, if the user says "hello," the system responds with "Hi there! How can I assist you?"

### 3. Project Idea 3: Travel Packing List Generator
- **Description:** A system that generates a packing list based on the user’s destination, climate, and trip duration.  
- **Rule-Based Approach:**  
  - The system uses rules to recommend items.  
  - For example, if the destination is "beach" and the climate is "hot," the system suggests sunscreen, swimsuits, and sunglasses.

### **Chosen Idea:** Recipe Recommender  
**Justification:** I chose this project because it is practical and applicable to real-life scenarios. It allows me to work with datasets, apply conditional logic, and create a system that provides meaningful recommendations based on user input.

---

## Part 2: Rules/Logic for the Chosen System

The **Recipe Recommender** system will follow these rules:

1. **Exact Match Rule:**  
   - **IF** all ingredients in a recipe are found in the user’s ingredient list → **Recommend the recipe.**

2. **Partial Match Rule:**  
   - **IF** 75% or more of the ingredients in a recipe match the user’s ingredient list →  
     - **Recommend the recipe.**  
     - **Suggest the missing ingredients.**

3. **Common Ingredients Rule:**  
   - Ingredients like salt, pepper, and water are considered optional and will not be counted as missing.

4. **No Match Rule:**  
   - **IF** no recipes match → **Suggest adding more ingredients** for better recommendations.

5. **Low Ingredient Rule:**  
   - **IF** fewer than three ingredients are provided → **Notify the user** and suggest adding more ingredients.

---

## Part 3: Rules/Logic for the Chosen System

Sample input and output: 

Enter your ingredients (comma-separated): chicken, rice, soy sauce
You are close to making Chicken Fried Rice! Missing: garlic.

Enter your ingredients (comma-separated): garlic, soy sauce
No recipes match. Try adding more ingredients.

Enter your ingredients (comma-separated): pasta, tomatoes, garlic, olive oil
You can make Spaghetti Pomodoro!

---

## Part 4: Reflection

### Project Overview:
This project involved designing a practical, rule-based system to recommend recipes based on user inputs. The system uses logical conditions (e.g., exact and partial matches) to evaluate user-provided ingredients against recipes in the dataset.

### Challenges:
- **Handling Partial Matches:**  
  Deciding on a threshold (75%) that balances flexibility with accuracy was challenging.
- **Common Ingredients:**  
  Ensuring common ingredients like salt and water don’t skew the results. I resolved this by excluding them from the missing ingredient list.















