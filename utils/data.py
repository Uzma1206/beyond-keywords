"""Sample documents and example queries used across the modules."""
from __future__ import annotations

# Module 1: the four documents from the brief
DEMO_DOCS = [
    {"title": "Tips to improve computer performance", "category": "Technology",
     "text": "Simple ways to get better results from your machine, from cleaning up files to upgrading parts."},
    {"title": "How to speed up a slow PC", "category": "Technology",
     "text": "Disable heavy startup programs and clear temporary files so your PC responds quickly again."},
    {"title": "Laptop battery maintenance guide", "category": "Technology",
     "text": "Avoid full discharges and extreme heat to help the battery last longer over the years."},
    {"title": "Understanding computer hardware", "category": "Technology",
     "text": "An introduction to the processor, memory, storage and the other parts inside a computer."},
]
DEMO_QUERY = "How can I make my laptop faster?"
DEMO_QUERIES = [
    DEMO_QUERY,
    "speed up slow PC",
    "computer performance tips",
    "laptop battery",
]

# Module 4: 12 documents, 6 categories
PLAYGROUND_DOCS = [
    {"title": "Fixing Python errors", "category": "Programming",
     "text": "Debugging is the process of finding and fixing mistakes in code. Read the error message, reproduce the problem, and use print statements or a debugger to trace where your program goes wrong."},
    {"title": "Writing readable code", "category": "Programming",
     "text": "Clean code uses meaningful variable names, short functions and helpful comments so that other developers can understand and maintain it easily."},
    {"title": "Effective study techniques", "category": "Education",
     "text": "Active recall and spaced repetition help students remember lessons for longer than rereading notes. Short, regular study sessions beat last-minute cramming before exams."},
    {"title": "Getting the most from lectures", "category": "Education",
     "text": "Take notes in your own words, ask questions during class, and review the material within 24 hours to understand difficult topics."},
    {"title": "Speeding up a sluggish computer", "category": "Technology",
     "text": "Close background programs, disable startup apps, free up disk space and consider an SSD upgrade so a slow computer responds quickly."},
    {"title": "Protecting your online accounts", "category": "Technology",
     "text": "Use strong unique passwords, turn on two-factor authentication and be careful with suspicious links to keep hackers out of your accounts."},
    {"title": "Sleep and student wellbeing", "category": "Health",
     "text": "Adults need seven to nine hours of sleep. A regular bedtime, less screen time at night and limited caffeine improve rest, mood and concentration."},
    {"title": "Staying active at a desk", "category": "Health",
     "text": "Stand up and stretch every hour, take short walks and do simple exercises to reduce back pain and fatigue from long hours of sitting."},
    {"title": "Preparing for a job interview", "category": "Career",
     "text": "Research the company, practise answers to common questions, and prepare examples that show your skills. Dress neatly and arrive early."},
    {"title": "Writing a strong resume", "category": "Career",
     "text": "Keep your CV to one page, highlight projects and achievements with measurable results, and tailor it to each job application."},
    {"title": "Budgeting for students", "category": "Finance",
     "text": "Track your monthly income and expenses, set spending limits, and save a small amount regularly so you are not short of money before the next allowance."},
    {"title": "Understanding investing basics", "category": "Finance",
     "text": "Investing means putting money into assets such as shares or mutual funds to grow wealth over time. Diversify and start early to benefit from compounding."},
]
PLAYGROUND_QUERY = "How do I make my laptop run faster?"
PLAYGROUND_EXAMPLES = [
    "How do I make my laptop run faster?",
    "I can't concentrate and feel tired all day",
    "My program keeps crashing, how do I find the cause?",
    "how to get hired after graduation",
    "ways to save money every month",
    "protect myself from hackers",
]
PLAYGROUND_CATEGORIES = ["Programming", "Education", "Technology", "Health", "Career", "Finance"]

# Module 5: fictional university knowledge base
UNIVERSITY_DOCS = [
    {"title": "Attendance Policy", "category": "Academics",
     "text": "Students must maintain at least 75% attendance in every course to be eligible for end-semester examinations. Students with medical emergencies can apply for condonation with a doctor's certificate submitted to the department office within one week of returning."},
    {"title": "Examination Guidelines", "category": "Academics",
     "text": "Candidates must carry a valid ID card and hall ticket to the examination hall. Mobile phones, smart watches and notes are not permitted. Entry is allowed up to 15 minutes after the exam starts."},
    {"title": "Library Timings", "category": "Campus",
     "text": "The central library is open from 8:00 AM to 10:00 PM on weekdays and 9:00 AM to 5:00 PM on weekends. During exam weeks the reading room stays open 24 hours. Students may borrow up to four books for 14 days."},
    {"title": "Scholarship Eligibility", "category": "Finance",
     "text": "Merit scholarships are available to students with a CGPA of 8.5 or above. Need-based financial aid requires a family annual income below 5 lakh rupees. Applications open every August through the student portal."},
    {"title": "Placement Registration Instructions", "category": "Career",
     "text": "Final-year students must register on the placement portal by September 30, upload an updated resume and verified marksheets. Registered students receive interview invitations from visiting companies."},
    {"title": "Hostel Rules", "category": "Campus",
     "text": "Hostel residents must return before 9:30 PM. Visitors are allowed in the common room between 4 PM and 6 PM only. Cooking appliances are not allowed in rooms."},
    {"title": "Fee Payment", "category": "Finance",
     "text": "Semester fees can be paid online through the student portal or at the accounts office. A late fee of 500 rupees per week applies after the due date. Installments can be requested with a written application to the registrar."},
    {"title": "Student Grievance Cell", "category": "Support",
     "text": "Students facing harassment, ragging or academic disputes can submit a complaint to the Grievance Cell by email or in person. Complaints are acknowledged within two days."},
]
UNIVERSITY_EXAMPLES = [
    "Can I sit for exams if I missed too many classes?",
    "When can I study in the library at night?",
    "Will I get financial help if my parents earn very little?",
    "What should I bring on exam day?",
    "How do I sign up to get hired by companies?",
    "I can't pay my fees on time, what happens?",
]
