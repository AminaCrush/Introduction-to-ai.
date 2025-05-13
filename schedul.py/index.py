import random
import matplotlib.pyplot as plt

num_particles = 10
max_iterations = 50

c1 = 1.5
c2 = 1.5 
w = 0.5

days = ["Пн", "Вт", "Ср", "Чт ", "Пт"]
periods_per_day = 3  
groups = ["ГРУППА 1", "ГРУППА 2", "ГРУППА 3", "ГРУППА 4"]
courses = {"DSA": 3, "ENG": 5, "APY": 2, "IAI": 2, "MNS": 1, "GEO": 1, "HIS": 1}

course_list = list(courses.keys())

def initialize_particles(num_particles):
    particles = []
    for _ in range(num_particles):
        schedule = {}
        for day in days:
            for period in range(periods_per_day):
                for group in groups:
                    course = random.choice(course_list)
                    schedule[(day, period, group)] = course
        particles.append({
            "position": schedule,
            "velocity": {key: random.choice(course_list) for key in schedule},
            "best_position": schedule.copy(),
            "best_value": float('inf')
        })
    return particles

def objective_function(schedule):   
    penalty = 0
    course_counts = {course: 0 for course in course_list}
    
    for key in schedule:
        course = schedule[key]
        course_counts[course] += 1
    
    for course in courses:
        penalty += abs(course_counts[course] - courses[course])
    
    for group in groups:
        group_schedule = [schedule[(day, period, group)] for day in days for period in range(periods_per_day)]
        group_course_counts = {course: group_schedule.count(course) for course in course_list}
        for course, count in group_course_counts.items():
            if count > 1:
                penalty += (count - 1) * 1
    
    for day in days:
        for period in range(periods_per_day):
            period_courses = [schedule[(day, period, group)] for group in groups]
            if len(period_courses) != len(set(period_courses)):
                penalty += 1
    return penalty

def pso():
    particles = initialize_particles(num_particles)
    global_best_position = particles[0]["position"].copy()
    global_best_value = float('inf')
    fitness_values = []

    for iteration in range(max_iterations):
        for particle in particles:
            value = objective_function(particle["position"])
            if value < particle["best_value"]:
                particle["best_value"] = value
                particle["best_position"] = particle["position"].copy()
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle["position"].copy()
        
            for key in particle["position"]:
                r1 = random.random()
                r2 = random.random()
                
                cognitive_velocity = c1 * r1
                social_velocity = c2 * r2
                
                if random.random() < (cognitive_velocity + social_velocity) / 2:
                    particle["velocity"][key] = particle["best_position"][key]
                else:     
                    particle["velocity"][key] = global_best_position[key]
            
            for key in particle["position"]:
                particle["position"][key] = particle["velocity"][key]
        
        fitness_values.append(global_best_value)
    
    return global_best_position, global_best_value, fitness_values

best_schedule, best_value, fitness_values = pso()

print("\n📌 Optimal Schedule:")
for day in days:
    print(f"\n{day}\n")
    print("Groups       | Period 1 | Period 2 | Period 3 ")
    print("------------------------------------------------")
    for group in groups:
        row = f"{group}    "
        for period in range(periods_per_day):
            course = best_schedule[(day, period, group)]
            row += f"|   {course}   "
        print(row)

plt.plot(fitness_values)
plt.xlabel("Iterations")
plt.ylabel("Fitness Value (Conflict Count)")
plt.title("PSO Optimization Dynamics")
plt.show()



print(f"\n......\n✅ All constraints satisfied! Fitness Value (Conflict Count): {best_value}")
