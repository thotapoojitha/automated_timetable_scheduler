from django.contrib import admin
from django.urls import path
from. import views

urlpatterns = [
    path('', views.home, name='home'),
    path('admin_login/', views.admin_login, name='admin_login'),
    path('student_login/', views.student_login, name='student_login'),
    path('select_course/', views.select_course, name='select_course'),
    path('student_home/', views.student_home, name='student_home'),
    path('resume_screening/', views.resume_screening, name='resume_screening'),
    path('calculate_cgpa/', views.calculate_cgpa, name='calculate_cgpa'),
    path('logout/', views.logout_view, name='logout'),
    path('view_timetable/', views.timetable, name='view_timetable'),
    path('login/',views.index,name="index"),
    path('Timetale/',views.admin_home,name="admin_home"),
    path('timetable_generation/', views.timetable, name='timetable'),
    path('add_room/', views.add_room, name='addroom'),
    path('add_instructor/', views.add_instructor, name='addinstructor'),
    path('instructor_list/', views.inst_list_view, name='editinstructor'),
    path('add_meetingtime/', views.add_meeting_time, name='addmeetingtime'),
    path('meetingtime_list/', views.meeting_list_view, name='editmeetingtime'),
    path('add_course/', views.add_course, name='addcourse'),
    path('course_list/', views.course_list_view, name='editcourse'),
    path('add_department/', views.add_department, name='adddepartment'),
    path('delete_meetingtime/<str:pk>/', views.delete_meeting_time, name='deletemeetingtime'),
    path('delete_course/<str:pk>/', views.delete_course, name='deletecourse'),
    path('delete_instructor/<int:pk>/', views.delete_instructor, name='deleteinstructor'),
    path('room_list/', views.room_list, name='editrooms'),
    path('delete_room/<int:pk>/', views.delete_room, name='deleteroom'),
    path('department_list/', views.department_list, name='editdepartment'),
    path('delete_department/<int:pk>/', views.delete_department, name='deletedepartment'),
    path('add_section/', views.add_section, name='addsection'),
    path('section_list/', views.section_list, name='editsection'),
    path('delete_section/<str:pk>/', views.delete_section, name='deletesection'),

]
