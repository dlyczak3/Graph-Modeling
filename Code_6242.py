#Libraries Used
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from matplotlib import image as mpimg
import math as m

#################################################     START OF CourseData Class     ########################################################################################################

class CourseData:
    def __init__(self, source_file):
        self.term_key = {} #Term key with keys as ordered term value and dictionary values being the term code(ie 201805)
        self.student_dictionary = {}#Dictionary with primary keys as student IDs, subkeys as terms they are enrolled, key values as courses enrolled
        self.term_enrollment = {} #dictionary with main keys are terms, subkeys as classes, and values as enrollment of that class for that given term
        self.ordered_courses_dictionary = {} #dictionary with keys as students, subkeys as ordered terms, and values as a list of courses taken
        self.enrollment_history = {} #Each courses overall enrollment count
        self.edge_tuple_list = [] #list of tuples (edge, node)
        self.node_tuple_list = [] #list of tuples (node, edge)
        self.edge_enrollment = {} #Count of class enrollment when the class is an edge
        self.node_enrollment = {} #Count of class enrollment when the class is an node
        self.edge_dictionary = {} #Dictionary with keys as (edge,node) groupings, and values as count of appearances
        self.node_dictionary = {} #Dictionary with keys as (node,edge) groupings, and values as count of appearances
        self.graph_dictionary = {} #Used for visualization only

        self.edge_weights = {} #results of the strength between edges and nodes
        self.node_weights = {} #results of the strength between nodes and edges
        self.source_file = source_file #required parameter for function to run


    def get_terms(self):
        infile = open(self.source_file , 'r')
        header = infile.readline()
        data = infile.readlines()
        infile.close()
        Term_set = set()
        for line in data:
            line_pieces = line.strip().split(',')
            TERM = line_pieces[1]
            Term_set.add(TERM)
        num_term = list(Term_set)
        num_term.sort()
        for i in range(len(num_term)):
            self.term_key[str(i)] = num_term[i] 


    def parse_data (self):
        infile = open(self.source_file , 'r')
        header = infile.readline()
        data = infile.readlines()
        infile.close()
        for line in data:
            line_pieces = line.strip().split(',')
            GTID = line_pieces[0]
            term_code = line_pieces[1]
            TRM = list(self.term_key.keys())[list(self.term_key.values()).index(line_pieces[1])]
            TRM = int(TRM)
            CRSE = str(line_pieces[2]) + str(line_pieces[3])
            if GTID not in self.student_dictionary:
                self.student_dictionary[GTID] = {}
                if TRM not in self.student_dictionary[GTID]:
                    self.student_dictionary[GTID][TRM] = []
                    self.student_dictionary[GTID][TRM].append(CRSE)
                else:
                    self.student_dictionary[GTID][TRM].append(CRSE)
            else:
                if TRM not in self.student_dictionary[GTID]:
                    self.student_dictionary[GTID][TRM] = []
                    self.student_dictionary[GTID][TRM].append(CRSE)
                else:
                    self.student_dictionary[GTID][TRM].append(CRSE)
            if term_code not in self.term_enrollment:
                self.term_enrollment[term_code] = {}
            elif CRSE not in self.term_enrollment[term_code]:
                self.term_enrollment[term_code][CRSE] = [] 
                self.term_enrollment[term_code][CRSE].append(GTID) 
            else:
                self.term_enrollment[term_code][CRSE].append(GTID)
            if CRSE not in self.enrollment_history:
                self.enrollment_history[CRSE] = 1
            else:
                self.enrollment_history[CRSE] += 1
        self.ordered_dictionary(self.student_dictionary)


    def ordered_dictionary(self, unsorted_dictionary):
        for i in unsorted_dictionary:
            sorted_dictionary = {key:value for key, value in sorted(unsorted_dictionary[i].items(), key = lambda item: int(item[0]))}
            self.ordered_courses_dictionary[i] = sorted_dictionary


    def edge_tuples(self):
        term_list = []
        for i in self.ordered_courses_dictionary:
            term_list = list(self.ordered_courses_dictionary[i].keys())
            for j in range(len(term_list)):
                k = j + 1
                for edge in self.ordered_courses_dictionary[i][term_list[j]]:
                    try:
                        for node in self.ordered_courses_dictionary[i][term_list[k]]:
                            edge_relation = (edge, node)
                            self.edge_tuple_list.append(edge_relation)
                    except:
                        continue
                    

    def node_tuples(self):
        term_list = []
        for i in self.ordered_courses_dictionary:
            term_list = list(self.ordered_courses_dictionary[i].keys())
            for j in range(len(term_list)):
                k = j + 1
                try:
                    for node in self.ordered_courses_dictionary[i][term_list[k]]:
                        for edge in self.ordered_courses_dictionary[i][term_list[j]]:
                            node_relation = (node, edge)
                            self.node_tuple_list.append(node_relation)
                except:
                    continue
                    

    def make_pair_counts(self, tuple_list, pair_dictionary):
        for i in tuple_list:
            if i not in pair_dictionary:
                pair_dictionary[i] = 1
            else:
                pair_dictionary[i] += 1


    def course_count(self, enroll_dict, edge = False):
        if edge:
            for i in self.ordered_courses_dictionary:
                term_list = list(self.ordered_courses_dictionary[i].keys())
                for j in range(len(term_list[:-1])):
                    for course in self.ordered_courses_dictionary[i][term_list[j]]:
                        if course not in enroll_dict:
                            enroll_dict[course] = [] 
                            enroll_dict[course].append(i)
                        else:
                            enroll_dict[course].append(i) 
        else:
            for i in self.ordered_courses_dictionary:
                term_list = list(self.ordered_courses_dictionary[i].keys())
                for j in range(len(term_list)): 
                    try:
                        for course in self.ordered_courses_dictionary[i][term_list[j+1]]:
                            if course not in enroll_dict:
                                enroll_dict[course] = [] 
                                enroll_dict[course].append(i)
                            else:
                                enroll_dict[course].append(i) #here too
                    except:
                        continue

                    
    def build_strength(self, pair_dictionary, enrollment_counts, strength_dictionary):
        for classes in pair_dictionary.keys():
            source = classes[0]
            target = classes[1]
            count = len(enrollment_counts[source]) 
            strength = (pair_dictionary[classes] / count)
            if source not in strength_dictionary:
                strength_dictionary[source] = {}
                strength_dictionary[source][target] = strength
            else:
                strength_dictionary[source][target] = strength


    def build_graph_dict(self):
        node_list = self.node_tuple_list.copy()
        for i in range(len(self.edge_tuple_list)):
            node_list.append(self.edge_tuple_list[i])
        for j in range(len(node_list)):
            n0 = node_list[j][0]
            n1 = node_list[j][1]
            tup0 = (n0, n1)
            tup1 = (n1, n0)
            if tup0 not in self.graph_dictionary:
                if tup1 not in self.graph_dictionary:
                    self.graph_dictionary[tup1] = 1
                else:
                    self.graph_dictionary[tup1] += 1
            else:
                self.graph_dictionary[tup0] += 1


    def run(self):
        self.get_terms()
        self.parse_data()
        self.edge_tuples()
        self.node_tuples() 
        self.make_pair_counts(self.edge_tuple_list, self.edge_dictionary)
        self.make_pair_counts(self.node_tuple_list, self.node_dictionary)
        self.course_count(enroll_dict = self.edge_enrollment, edge = True)
        self.course_count(enroll_dict = self.node_enrollment, edge = False)
        self.build_strength(self.edge_dictionary, self.edge_enrollment, self.edge_weights)
        self.build_strength(self.node_dictionary, self.node_enrollment, self.node_weights)
        self.build_graph_dict()

#################################################     END OF CourseData Class     ########################################################################################################

#################################################     START OF Predictive Functions Called in Notebook     ########################################################################################################

def load_data(File_Path):
    if len(File_Path) < 1:
        cd = CourseData('course_history.csv')
        cd.run()
        print('Your Data has been successfully loaded!')
        return cd
    elif File_Path[-3:] == 'csv':
        cd = CourseData(str(File_Path))
        try:
            cd.run()
            print('Your Data has been successfully loaded!')
            return cd
        except:
            print('there was an error in loading your data. Please ensure your FilePath is correct, and that it is in the form of a CSV file')
    else:
        print('The file you selected is either in a different location, or not a CSV file. If you would like to use the default data provided to you, please put "" for File_Path')

def PredictEnrollment(terms, enrollment_history, course_strength, course):
    level = int([i for i in course if i.isnumeric()][0])
    weight = 0
    if level == 1:
        weight += 0.3
    elif level == 2:
        weight += 0.25
    else:
        weight += 0.2
    term_List = []
    for k,v in terms.items():
        term_List.append(int(k))
    term_List = sorted(term_List)
    target_term = terms[str(term_List[-1])]
    historic_enrollment = enrollment_history[target_term]
    course_flow = course_strength[str(course)]
    potential_students = {}
    predicted_enrollment = 0
    for paired, strength in course_flow.items():
        if paired in historic_enrollment.keys() and (strength > weight):
            for i in historic_enrollment[paired]:
                if i not in historic_enrollment[course]:
                    if i not in potential_students:
                        potential_students[i] = []
                        potential_students[i].append(strength)
                    else:
                        potential_students[i].append(strength)
                else:
                    continue
        else:
            continue
    for ID, prob in potential_students.items():
        #X = (sum(prob)) * len(prob) - (weight)
        X = (sum(prob)) - (weight)
        likelihood = m.exp(X) / (m.exp(X) + 1)
        predicted_enrollment += likelihood
    if (int(predicted_enrollment) > len(historic_enrollment[course])):
        fig=plt.figure(figsize=(12,4), dpi= 100, facecolor='#F9F6E5', edgecolor='#F9F6E5')
        plt.title("Enrollment for {} is expected to increase".format(course))
        image = mpimg.imread("increase.jpg")
        plt.axis('off')
        plt.imshow(image)
        return plt.show()
        #return 'increase' , int(predicted_enrollment)
    else:
        fig=plt.figure(figsize=(12,4), dpi= 100, facecolor='#F9F6E5', edgecolor='#F9F6E5')
        plt.title("Enrollment for {} is expected to decline".format(course))
        image = mpimg.imread("decrease.jpg")
        plt.axis('off')
        plt.imshow(image)
        return plt.show()


def MatchStudent (GTID,enrollment_history):
    current_courses = set(enrollment_history[GTID][max(enrollment_history[GTID].keys())])
    past_enrollment = set()
    similar_students = {}
    course_repository = {}
    course_repository['student_pool'] = set()
    course_recommendation = {}
    for term, courses in enrollment_history[GTID].items():
        for course in courses:
            past_enrollment.add(course)
    for student in enrollment_history:
        if student != GTID:
            term_list = sorted(list(enrollment_history[student].keys()))
            #recent = term_list[-1] 
            for i in range(len(term_list[:-1])):
                if (set(enrollment_history[student][term_list[i]]) >= current_courses) or (current_courses >= set(enrollment_history[student][term_list[i]])) or (current_courses == set(enrollment_history[student][term_list[i]])):
                    similar_students[student] = int(term_list[i-1])
                else:
                    continue                    
        else:
            continue
    if len(similar_students) < 1:
        return 'No Matches Found'
    else:
        for student, target_term in similar_students.items():
            course_list = enrollment_history[student][target_term]
            for course in course_list:
                if course not in past_enrollment:
                    course_repository['student_pool'].add(student)
                    if course not in course_repository:
                        course_repository[course] = 1
                    else:
                        course_repository[course] += 1
                else:
                    continue
    if len(course_repository['student_pool']) > 0:
        for course, count in course_repository.items():
            student_count = len(course_repository['student_pool'])
            if course != 'student_pool':
                course_recommendation[course] = (count / student_count)
            else:
                continue
    else:
        return "No Matches Found"
    return course_recommendation

#################################################     END OF Predictive Functions Called in Notebook     ########################################################################################################

#################################################     START OF Visualization (sans Graph) Functions     ########################################################################################################

def CourseLollipop (history, Course):
    viz_dict = CourseViz(history, Course)
    if len(viz_dict) > 0:

        fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='#F9F6E5', edgecolor='#F9F6E5')
        font1 = {'family':'monospace','color':'#54585A','size':27}
        font2 = {'family':'monospace','color':'#EAAA00','size':23}

        values= list(viz_dict.values())
        names = list(viz_dict.keys())

        plt.title("Historical Enrollment of {}".format(Course), loc='left', fontdict = font1)
        plt.ylabel('Total Enrollment', fontdict = font2)
        plt.xlabel('Term', fontdict = font2)
        (markers, stemlines, baseline) = plt.stem(names, values)
        plt.setp(stemlines, linestyle="-", color='#B3A369', linewidth=10)
        plt.setp(markers, marker='D', markersize=7, markeredgecolor="#003057", markeredgewidth=7)
        plt.setp(baseline, linestyle="--", color="black", linewidth=1)
        plt.ylim(0, max(values) + 50)

        for i, v in enumerate(values):
            plt.annotate(str(v), xy=(i,v), xytext=(7,7), textcoords='offset pixels', color = '#003057')

        return plt.show()
    else:
        return print("No course found. Please ensure that you're providing a valid course in the format of Subject + course number, example 'CS1301'.")


def StudentViz(term_keys, history):
    term_dict = {}
    viz_dict = {}
    for order, term in term_keys.items():
        if term[4:] == '02':
            sem = 'Spring'
        elif term[4:] == '05':
            sem = 'Summer'
        elif term[4:] == '08':
            sem = 'Fall'
        else:
            sem = 'Other'
        term_dict[int(order)] = "{} {}".format(sem, int(term[:-2]))
    for term, courses in history.items():
        viz_dict[term_dict[term]] = np.array(courses)
    return viz_dict


def CourseViz(history, Course):
    temp_list = []
    viz_dict = {}
    for term, course in history.items():
        if Course in history[term].keys():
            new_tup = (int(term), len(history[term][Course]))
            temp_list.append(new_tup)
    temp_list = sorted(temp_list, key = lambda tup: tup[0])
    for term, enrollment in temp_list:
        viz_dict[str(term)] = int(enrollment)
    return viz_dict


def StudentEnrollment(GTID, term_key, history):
    try:
        past = StudentViz(term_key, history[GTID])
        future = MatchStudent(GTID, history)
        df_future = pd.DataFrame.from_dict(list(future.items()))
        df_future = df_future.nlargest(n=10, columns=[1])
        df_future[1] = df_future[1].apply(lambda x: round(x * 100, 2))
        df_future.reset_index(drop = True, inplace = True)
        df_future = df_future.rename(columns = {0: 'Recommended Class', 1: 'Recommendation Strength'})
        df_future[' '] = '-->'
        df_future = df_future[[' ', 'Recommended Class' , 'Recommendation Strength']]
        df_past = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in past.items()]))
        df_final = pd.concat([df_past,df_future],axis=1,sort=False)
        df_final.fillna(value = '  -  ', inplace = True)
        return df_final
    except:
        try:
            past = StudentViz(term_key, history[GTID])
            df_past = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in past.items()]))
            df_past.fillna(value = '  -  ', inplace = True)
            print("No students matched the most recent enrollment")
            return df_past
        except:
            return print ('An error has occured. Please make sure you are providing a valid Student ID, and that the student has enrolled in at least one semester')

#################################################     END OF Visualization (sans Graph) Functions     ########################################################################################################

#################################################     START OF Graph Visualization Functions     ########################################################################################################

def AddNodes (target_list, graph_dictionary):
    temp_list = []
    return_list = []
    for i in range(len(target_list)):
        course = target_list[i]
        for j in range(len(target_list)):
            if course != target_list[j]:
                try:
                    tup0 = (course, target_list[j])
                    tup1 = (target_list[j], course)
                    if tup0 in graph_dictionary:
                        if tup0 not in temp_list:
                            if tup1 not in temp_list:
                                temp_list.append(tup0)
                    elif tup1 in graph_dictionary:
                        if tup1 not in temp_list:
                            if tup0 not in temp_list:
                                temp_list.append(tup1)
                except:
                    continue
            else:
                continue
    for k in range(len(temp_list)):
        n0 = temp_list[k][0]
        n1 = temp_list[k][1]
        if (n0, n1) in graph_dictionary:
            weight = graph_dictionary[(n0, n1)]
            return_list.append((n0, n1, weight))
        elif (n1, n0) in graph_dictionary:
            weight = graph_dictionary[(n1, n0)]
            return_list.append((n1, n0, weight))
        else:
            continue
    return return_list


def CourseGraph(graph_dictionary, g_target, size):
    connections_dict = {}
    graph_nodes = []
    for i in graph_dictionary.keys():
        if i[0] == g_target:
            connections_dict[i[1]] = graph_dictionary[i]
        elif i[1] == g_target:
            connections_dict[i[0]] = graph_dictionary[i]
        else:
            continue
    df1 = pd.DataFrame.from_dict(list(connections_dict.items()))
    df2 = df1.nlargest(n=size, columns=[1])
    paired = list(df2[0])
    p_strength = list(df2[1])
    additional_connections = AddNodes(paired, graph_dictionary)
    for i in range(len(additional_connections)):
        graph_nodes.append(additional_connections[i])
    for k in range(len(paired)):
        new_tup = (g_target, paired[k], p_strength[k])
        graph_nodes.append(new_tup)
    return graph_nodes

def CourseHistory(nodes, history):
    course_set = set()
    enrollment_dictionary = {}
    for i in range(len(nodes)):
        n0 = nodes[i][0]
        n1 = nodes[i][1]
        course_set.add(n0)
        course_set.add(n1)
    for j in course_set:
        stu_count = history[j]
        enrollment_dictionary[j] = stu_count
    return enrollment_dictionary

def GenerateGraph(Course, enrollment_history, graph_dictionary, size):
    nodes = CourseGraph(graph_dictionary, Course, size)
    headcount = CourseHistory(nodes, enrollment_history)
    #####Works up to here###########
    proportion = (size * 0.1)
    df = pd.DataFrame(nodes, columns = ['n1', 'n2', 'weight'])
    G = nx.from_pandas_edgelist(df, 'n1', 'n2', edge_attr ='weight')
    for i in list(G.nodes()):
        G.nodes[i]['enrollment'] = headcount[i]
    plt.figure(figsize =(20, min(12,(size * 2))), dpi= 100, facecolor='#F9F6E5', edgecolor='#F9F6E5')
    node_color = [G.degree(v) for v in G]
    node_size = [(2 + proportion) * nx.get_node_attributes(G, 'enrollment')[v] for v in G]
    edge_width = [(G[u][v]['weight'] * 0.003) for u, v in G.edges()]
    plt.axis('off')
    nx.draw_networkx(G, pos = nx.circular_layout(G), node_color = node_color, alpha = 0.7, edge_color ='.7', cmap = plt.cm.cividis, width = edge_width, node_size = node_size)
    return plt.show()

#################################################     END OF Graph Visualization Functions     ########################################################################################################

def greetings():
    print('Hi TAs, I know that being a TA is a thankless job, so I just wanted to thank you for this course, as its challenges ultimately made me a stronger analyst. Thank you')
