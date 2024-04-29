import numpy
import cv2
import matplotlib.pyplot as plt

# Function to generate city coordinates
def Generate(width, height, count):
    # Dictionary of Indian city coordinates
    indian_cities = {
        "Mumbai": (16.0760, 72.8777),
        "Delhi": (28.7041, 77.1025),
        "Goa": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
        "Hyderabad": (17.3850, 78.4867),
        "Ahmedabad": (18.0225, 80.5714),
        "Pune": (18.5204, 74.9567),
        "Surat": (21.1702, 72.8311),
        "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 87.9462),
        "Kanpur": (26.4499, 80.3319),
        "Nagpur": (21.1458, 79.0882),
        "Patna": (25.5941, 85.1376),
        "Indore": (22.7196, 75.8577)
    }
    city_names = list(indian_cities.keys())
    numpy.random.shuffle(city_names)  # Shuffle city names to ensure randomness

    # Calculate the average distance between cities
    avg_distance_x = width / (count * 1.5)  # Adjusted for proper spacing
    avg_distance_y = height / (count * 1.5)  # Adjusted for proper spacing

    cities = []
    for i in range(count):
        city_name = city_names[i]
        latitude, longitude = indian_cities[city_name]
        # Convert latitude and longitude to pixel coordinates
        pixel_x = longitude * 27 - 1700
        pixel_y = latitude * 23 - 100
        cities.append((int(pixel_x), int(pixel_y), city_name))
    return cities

# Function to initialize the solution
def Initialize(count):
    solution = numpy.arange(count)
    numpy.random.shuffle(solution)
    return solution

# Function to evaluate the solution
def Evaluate(cities, solution):
    distance = 0
    for i in range(len(cities)):
        index_a = solution[i]
        index_b = solution[i - 1]
        delta_x = cities[index_a][0] - cities[index_b][0]
        delta_y = cities[index_a][1] - cities[index_b][1]
        distance += (delta_x ** 2 + delta_y ** 2) ** 0.5
    return distance

# Function to evaluate the solution in kilometers
def Evaluate_km(cities, solution, scale):
    distance_px = Evaluate(cities, solution)
    distance_km = distance_px * scale
    return distance_km

# Function to modify the solution
def Modify(current):
    new = current.copy()
    index_a = numpy.random.randint(len(current))
    index_b = numpy.random.randint(len(current))
    while index_b == index_a:
        index_b = numpy.random.randint(len(current))
    new[index_a], new[index_b] = new[index_b], new[index_a]
    return new

# Function to draw the visualization
def Draw(width, height, cities, solution, infos, iteration):
    frame = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    for i in range(len(cities)):
        index_a = solution[i]
        index_b = solution[i - 1]
        point_a = (cities[index_a][0], cities[index_a][1])
        point_b = (cities[index_b][0], cities[index_b][1])
        cv2.line(frame, point_a, point_b, (0, 255, 0), 2)
    for city in cities:
        cv2.circle(frame, (city[0], city[1]), 5, (0, 0, 255), -1)
        cv2.putText(frame, city[2], (city[0] + 10, city[1] + 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))

    # Draw other information
    cv2.putText(frame, f"Score", (25, 75), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f"Best Score", (25, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f"Worst Score", (25, 125), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f"Temperature", (25, 150), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f"Iteration", (25, 175), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f": {infos[1]:.2f}", (175, 75), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f": {infos[2]:.2f}", (175, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f": {infos[3]:.2f}", (175, 125), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f": {infos[0]:.2f}", (175, 150), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
    cv2.putText(frame, f": {iteration}", (175, 175), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))

    cv2.imshow("Simulated Annealing", frame)
    cv2.waitKey(5)

# Function to plot temperature variation
def PlotTemperature(temperature_history):
    plt.figure(figsize=(8, 4))
    plt.plot(temperature_history, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature Variation')
    plt.savefig('temperature_graph.png')

# Function to plot score variation
def PlotScore(score_history, best_score_history, worst_score_history):
    plt.figure(figsize=(8, 6))
    plt.plot(score_history, label='Score', color='blue')
    plt.plot(best_score_history, label='Best Score', color='green')
    plt.plot(worst_score_history, label='Worst Score', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score Variation')
    plt.legend()
    plt.savefig('score_variation_graph.png')

# Constants
WIDTH = 800  # Width of the map in pixels
HEIGHT = 640  # Height of the map in pixels
CITY_COUNT = 15
INITIAL_TEMPERATURE = 1000
STOPPING_TEMPERATURE = 1
TEMPERATURE_DECAY = 0.999
SCALE = 10  # Hypothetical scale: 1 pixel = 10 km

if __name__ == "__main__":
    cities = Generate(WIDTH, HEIGHT, CITY_COUNT)
    current_solution = Initialize(CITY_COUNT)
    current_score = Evaluate_km(cities, current_solution, SCALE)
    best_score = worst_score = current_score
    temperature = INITIAL_TEMPERATURE
    temperature_history = []
    score_history = []
    best_score_history = []
    worst_score_history = []
    iteration = 0

    # Main loop
    while temperature > STOPPING_TEMPERATURE:
        new_solution = Modify(current_solution)
        new_score = Evaluate_km(cities, new_solution, SCALE)
        best_score = min(best_score, new_score)
        worst_score = max(worst_score, new_score)
        if new_score < current_score:
            current_solution = new_solution
            current_score = new_score
        else:
            delta = new_score - current_score
            probability = numpy.exp(-delta / temperature)
            if probability > numpy.random.uniform():
                current_solution = new_solution
                current_score = new_score
        # Update temperature and history
        temperature *= TEMPERATURE_DECAY
        temperature_history.append(temperature)
        score_history.append(current_score)
        best_score_history.append(best_score)
        worst_score_history.append(worst_score)
        infos = (temperature, current_score, best_score, worst_score)
        iteration += 1
        Draw(WIDTH, HEIGHT, cities, current_solution, infos, iteration)

    # Plotting graphs
    PlotTemperature(temperature_history)
    PlotScore(score_history, best_score_history, worst_score_history)

    # Retrieving final list of cities
    final_cities = [cities[index][2] for index in current_solution]
    print("Final list of cities visited:")
    print(final_cities)

    # Calculating and printing the final distance in kilometers
    final_distance = Evaluate_km(cities, current_solution, SCALE)
    print("Final distance traveled (km):", final_distance)
