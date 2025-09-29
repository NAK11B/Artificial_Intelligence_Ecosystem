
# Dataset of recipes
recipes = {
    "Spaghetti Pomodoro": ["pasta", "tomatoes", "garlic", "olive oil"],
    "Chicken Fried Rice": ["chicken", "rice", "garlic", "soy sauce"],
    "Pancakes": ["flour", "milk", "eggs", "sugar"],
    "Vegetable Stir-Fry": ["broccoli", "carrots", "soy sauce", "garlic", "ginger"],
    "Caprese Salad": ["tomatoes", "mozzarella", "basil", "olive oil", "balsamic vinegar"],
    "Banana Smoothie": ["banana", "milk", "yogurt", "honey", "ice"],
    "Beef Burrito": ["tortilla", "ground beef", "cheese", "lettuce", "tomato"],
    "Turkey Sandwich": ["bread", "turkey", "cheese", "lettuce", "tomato"],
    "Chicken Pot Pie": ["pie crust", "chicken", "cream of mushroom soup", "carrots", "green beans"],
    "Ramen": ["noodles", "broth", "vegetables", "hard boiled egg", "soy sauce"],
    "Salsa": ["tomatoes", "garlic", "onion", "jalepenos", "cilantro"],
    "Chef's Salad": ["lettuce","tomatoes", "ham", "turkey", "hard boiled egg", "salad dressing"],
    "Grilled Cheese Sandiwich": ['bread', 'cheese', 'mayonaise', 'pickles'],
    "Bread": ['flour', 'yeast', 'oil']
}

# Function to recommend recipes
def recommend_recipes(user_ingredients):
    recommendations = []
    user_ingredients = [ingredient.lower().strip() for ingredient in user_ingredients]
    for recipe, ingredients in recipes.items():
        common = set(user_ingredients).intersection(ingredients)
        missing = set(ingredients) - set(user_ingredients)
        if len(common) == len(ingredients):  # Exact match
            recommendations.append(f"You can make {recipe}!")
        elif len(common) / len(ingredients) >= 0.75:  # Partial match
            recommendations.append(f"You are close to making {recipe}! Missing: {', '.join(missing)}.")
    return recommendations if recommendations else ["No recipes match. Try adding more ingredients."]

# Interactive input loop
print("Welcome to the Recipe Recommender!")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nEnter your ingredients (comma-separated): ")
    if user_input.lower() == "exit":
        print("Goodbye! Happy cooking!")
        break
    user_ingredients = user_input.split(",")
    recommendations = recommend_recipes(user_ingredients)
    for rec in recommendations:
        print(rec)