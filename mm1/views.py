from django.shortcuts import render, redirect
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Room, MeetingTime, Instructor, Course, Department, Section
from .forms import InstructorForm, RoomForm, MeetingTimeForm, CourseForm, DepartmentForm, SectionForm
import random as rnd
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Room, MeetingTime, Instructor, Course, Department, Section
from .forms import InstructorForm, RoomForm, MeetingTimeForm, CourseForm, DepartmentForm, SectionForm
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
import random as rnd
import os
import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render
import os
import pickle
from django.conf import settings
# views.py

import pickle
import re
import nltk
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Load necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')





pickle_files_dir = os.path.join(settings.BASE_DIR, 'data')
with open(os.path.join(pickle_files_dir, 'clf.pkl'), 'rb') as f:
    clf = pickle.load(f)

with open(os.path.join(pickle_files_dir, 'tfidf.pkl'), 'rb') as f:
    tfidfd = pickle.load(f)
# Load pickle files
with open(os.path.join(pickle_files_dir, 'courses.pkl'), 'rb') as f:
    courses_list = pickle.load(f)

with open(os.path.join(pickle_files_dir, 'similarity.pkl'), 'rb') as f:
    similarity = pickle.load(f)

# Function to clean resume text
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# View function to handle resume screening
@csrf_exempt  # Remove if CSRF protection is not needed
def resume_screening(request):
    if request.method == 'POST':
        if 'resume' in request.FILES:
            uploaded_file = request.FILES['resume']
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

            # Clean the resume text
            cleaned_resume = clean_resume(resume_text)

            # Transform the resume text using tfidf
            input_features = tfidfd.transform([cleaned_resume])

            # Make the prediction using the loaded classifier
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            # Render to result page
            return render(request, 'result.html', {'category_name': category_name})
        else:
            return render(request, 'screening.html', {'error': 'No file uploaded'})
    else:
        return render(request, 'screening.html')
def recommend(course):
    index = courses_list[courses_list['course_name'] == course].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_course_names = []
    for i in distances[1:7]:
        course_name = courses_list.iloc[i[0]].course_name
        recommended_course_names.append(course_name)
    return recommended_course_names

def select_course(request):
    course_list = courses_list['course_name'].values
    recommended_courses = []
    
    if request.method == 'POST':
        selected_course = request.POST.get('selected_course')
        recommended_courses = recommend(selected_course)
    
    context = {
        'course_list': course_list,
        'recommended_courses': recommended_courses,
    }
    return render(request, 'select_course.html', context)
# Constants
POPULATION_SIZE = 9
NUMB_OF_ELITE_SCHEDULES = 1
TOURNAMENT_SELECTION_SIZE = 3
MUTATION_RATE = 0.05

MAX_GENERATIONS = 50  # Set a maximum number of generations

# Data class to fetch data from the database
class Data:
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()

    def get_rooms(self): return self._rooms
    def get_instructors(self): return self._instructors
    def get_courses(self): return self._courses
    def get_depts(self): return self._depts
    def get_meetingTimes(self): return self._meetingTimes

data = Data()

class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._classNumb = 0
        self._isFitnessChanged = True

    def get_classes(self):
        self._isFitnessChanged = True
        return self._classes

    def get_numbOfConflicts(self): return self._numberOfConflicts

    def get_fitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculate_fitness()
            self._isFitnessChanged = False
        return self._fitness

    def initialize(self):
        sections = Section.objects.all()
        for section in sections:
            dept = section.department
            n = section.num_class_in_week
            if n <= len(MeetingTime.objects.all()):
                courses = dept.courses.all()
                for course in courses:
                    for i in range(n // len(courses)):
                        crs_inst = course.instructors.all()
                        newClass = Class(self._classNumb, dept, section.section_id, course)
                        self._classNumb += 1
                        newClass.set_meetingTime(data.get_meetingTimes()[rnd.randrange(0, len(MeetingTime.objects.all()))])
                        newClass.set_room(data.get_rooms()[rnd.randrange(0, len(data.get_rooms()))])
                        newClass.set_instructor(crs_inst[rnd.randrange(0, len(crs_inst))])
                        self._classes.append(newClass)
            else:
                n = len(MeetingTime.objects.all())
                courses = dept.courses.all()
                for course in courses:
                    for i in range(n // len(courses)):
                        crs_inst = course.instructors.all()
                        newClass = Class(self._classNumb, dept, section.section_id, course)
                        self._classNumb += 1
                        newClass.set_meetingTime(data.get_meetingTimes()[rnd.randrange(0, len(MeetingTime.objects.all()))])
                        newClass.set_room(data.get_rooms()[rnd.randrange(0, len(data.get_rooms()))])
                        newClass.set_instructor(crs_inst[rnd.randrange(0, len(crs_inst))])
                        self._classes.append(newClass)

        return self

    def calculate_fitness(self):
        self._numberOfConflicts = 0
        classes = self.get_classes()
        for i in range(len(classes)):
            if classes[i].room.seating_capacity < int(classes[i].course.max_numb_students):
                self._numberOfConflicts += 1
            for j in range(len(classes)):
                if j >= i:
                    if (classes[i].meeting_time == classes[j].meeting_time) and \
                            (classes[i].section_id != classes[j].section_id) and (classes[i].section == classes[j].section):
                        if classes[i].room == classes[j].room:
                            self._numberOfConflicts += 1
                        if classes[i].instructor == classes[j].instructor:
                            self._numberOfConflicts += 1
        return 1 / (1.0 * self._numberOfConflicts + 1)

class Population:
    def __init__(self, size):
        self._size = size
        self._data = data
        self._schedules = [Schedule().initialize() for i in range(size)]

    def get_schedules(self):
        return self._schedules

class GeneticAlgorithm:
    def evolve(self, population):
        return self._mutate_population(self._crossover_population(population))

    def _crossover_population(self, pop):
        crossover_pop = Population(0)
        for i in range(NUMB_OF_ELITE_SCHEDULES):
            crossover_pop.get_schedules().append(pop.get_schedules()[i])
        i = NUMB_OF_ELITE_SCHEDULES
        while i < POPULATION_SIZE:
            schedule1 = self._select_tournament_population(pop).get_schedules()[0]
            schedule2 = self._select_tournament_population(pop).get_schedules()[0]
            crossover_pop.get_schedules().append(self._crossover_schedule(schedule1, schedule2))
            i += 1
        return crossover_pop

    def _mutate_population(self, population):
        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            self._mutate_schedule(population.get_schedules()[i])
        return population

    def _crossover_schedule(self, schedule1, schedule2):
        crossoverSchedule = Schedule().initialize()
        for i in range(0, len(crossoverSchedule.get_classes())):
            if rnd.random() > 0.5:
                crossoverSchedule.get_classes()[i] = schedule1.get_classes()[i]
            else:
                crossoverSchedule.get_classes()[i] = schedule2.get_classes()[i]
        return crossoverSchedule

    def _mutate_schedule(self, mutateSchedule):
        schedule = Schedule().initialize()
        for i in range(len(mutateSchedule.get_classes())):
            if MUTATION_RATE > rnd.random():
                mutateSchedule.get_classes()[i] = schedule.get_classes()[i]
        return mutateSchedule

    def _select_tournament_population(self, pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_schedules().append(pop.get_schedules()[rnd.randrange(0, POPULATION_SIZE)])
            i += 1
        tournament_pop.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop

class Class:
    def __init__(self, id, dept, section, course):
        self.section_id = id
        self.department = dept
        self.course = course
        self.instructor = None
        self.meeting_time = None
        self.room = None
        self.section = section

    def get_id(self): return self.section_id
    def get_dept(self): return self.department
    def get_course(self): return self.course
    def get_instructor(self): return self.instructor
    def get_meetingTime(self): return self.meeting_time
    def get_room(self): return self.room

    def set_instructor(self, instructor): self.instructor = instructor
    def set_meetingTime(self, meetingTime): self.meeting_time = meetingTime
    def set_room(self, room): self.room = room

def context_manager(schedule):
    classes = schedule.get_classes()
    context = []
    for i in range(len(classes)):
        cls = {}
        cls["section"] = classes[i].section_id
        cls['dept'] = classes[i].department.dept_name
        cls['course'] = f'{classes[i].course.course_name} ({classes[i].course.course_number}, ' \
                        f'{classes[i].course.max_numb_students}'
        cls['room'] = f'{classes[i].room.r_number} ({classes[i].room.seating_capacity})'
        cls['instructor'] = f'{classes[i].instructor.name} ({classes[i].instructor.uid})'
        cls['meeting_time'] = [classes[i].meeting_time.pid, classes[i].meeting_time.day, classes[i].meeting_time.time]
        context.append(cls)
    return context

def home(request):
    return render(request, 'homepage.html', {})

def admin_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            auth_login(request, user)
            return redirect('index')  
        else:
            return render(request, 'admin_login.html', {'error_message': 'Invalid credentials'})
    return render(request, 'admin_login.html')
def student_login(request):
    error_message = ''
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        # Define static credentials
        valid_username = 'student'
        valid_password = 'password123'
        if username == valid_username and password == valid_password:
            request.session['student_username'] = username
            return redirect('student_home')
        else:
            error_message = 'Invalid credentials. Please try again.'
    return render(request, 'student_login.html', {'error_message': error_message})

def student_home(request):
    return render(request, 'student_home.html')
def index(request):
    return render(request, 'index.html')
def admin_home(request):
    return render(request,'index.html')
    
def timetable(request):
    schedule = []
    population = Population(POPULATION_SIZE)
    generation_num = 0
    population.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
    geneticAlgorithm = GeneticAlgorithm()
    
    while population.get_schedules()[0].get_fitness() != 1.0 and generation_num < MAX_GENERATIONS:
        generation_num += 1
        print('\n> Generation #' + str(generation_num))
        population = geneticAlgorithm.evolve(population)
        population.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
        schedule = population.get_schedules()[0].get_classes()

    return render(request, 'timetable.html', {'schedule': schedule, 'sections': Section.objects.all(),
                                              'times': MeetingTime.objects.all()})

def add_instructor(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addinstructor')
    context = {'form': form}
    return render(request, 'adins.html', context)

def inst_list_view(request):
    context = {'instructors': Instructor.objects.all()}
    return render(request, 'instlist.html', context)

def delete_instructor(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('editinstructor')

def add_room(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addroom')
    context = {'form': form}
    return render(request, 'addrm.html', context)

def room_list(request):
    context = {'rooms': Room.objects.all()}
    return render(request, 'rmlist.html', context)

def delete_room(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('editrooms')

def meeting_list_view(request):
    context = {'meeting_times': MeetingTime.objects.all()}
    return render(request, 'mtlist.html', context)

def add_meeting_time(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addmeetingtime')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'addmt.html', context)

def delete_meeting_time(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('editmeetingtime')

def course_list_view(request):
    context = {'courses': Course.objects.all()}
    return render(request, 'crslist.html', context)

def add_course(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addcourse')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'adcrs.html', context)

def delete_course(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('editcourse')

def add_department(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('adddepartment')
    context = {'form': form}
    return render(request, 'addep.html', context)

def department_list(request):
    context = {'departments': Department.objects.all()}
    return render(request, 'deptlist.html', context)

def delete_department(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('editdepartment')

def calculate_cgpa(request):
    return render(request, 'calculate_cgpa.html')

def add_section(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addsection')
    context = {'form': form}
    return render(request, 'addsec.html', context)

def section_list(request):
    context = {'sections': Section.objects.all()}
    return render(request, 'seclist.html', context)

def delete_section(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('editsection')
def logout_view(request):
    # Your logout logic here
    return redirect('home')  