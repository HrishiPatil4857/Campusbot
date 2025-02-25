import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter 

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=16384, chunk_overlap=1024)
loader = Docx2txtLoader('About_the_founders.docx')
data = loader.load()
doc_splits = text_splitter.split_documents(data)

from langchain_community.embeddings import OllamaEmbeddings


vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.partial_output = ""

    def on_llm_new_token(self, token, **kwargs):
        self.partial_output += token
        print(token, end="", flush=True)

 # URL processing
def process_input(question):
    model_local = Ollama(model="llama2", callbacks=[StreamingCallbackHandler()])

    after_rag_template = """ You are a college chatbot, you give answers to queries for college. Your answers are solely based on facts. Your answer doesn't contain the question. If you are asked for all faculties, you return all of the faculties available in your context.Whenever mentioning faculty name also mention his/her qualifications and position Your answers don't contain useless information. You answer only what is asked. You always recheck your answer twice to make sure if it is answering the question asked correctly .Answer the question given based only on the following context:{context}    
    {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    question = f'Question : {question} '
    question = question.lower()
    if 'principal' in question or 'uday' in question or 'principle' in question:
        question = """Dr. Udaykumar Kalyane  , Principal
        As the Principle of BKIT  also known as t Bheemanna Khandre Institute of Technology, Bhalki  Established in the year 1982, I am very much delighted to share with you all that Bheemanna Khandre Institute of Technology, Bhalki is one of the premier engineering college of the region. Certainly it has produced 6500 plus eminent engineers and technocrats of whom some are currently serving the nation in various capacities.
        """ + question
    elif ('intake' in question and 'procedure' not in question ) :
        question = """UG Courses\nSr. No.\tBranches\tIntake\n1\tCivil Engineering\t60+3*\n2\tMechanical Engineering\t60+3*\n3\tElectronics & Communication Engineering\t90+5*\n4\tComputer Science and Engineering\t120+6*\n5\tChemical Engineering\t30+2*\n6\tCSE (Artificial Intelligence & Machine Learning)\t60+3*\n7\tCSE (Data Science)\t60+3*\n8\tCSE (Cyber Securuty)\t60+3*\n\nPG Courses\nSr. No.\tBranches\tIntake\n1\tM.Tech in Computer Science and Engineering\t18\n2\tM.Tech in Geotechnical Engineering (under Civil Engg.)\t18\n3\tMCA (Master of Computer Application)\t60\n\nPh. D. at Research Centres of\nElectronics & Communication Engineering\nMechanical Engineering\nPhysics\nChemistry\nMathematics\nCivil Engineering\nComputer Science and Engineering\n\nFor Admission or Further details Please Contact:\nSri Vijaykumar Bardapure, Admission Superintendent    Mobile:  09448459916\nDr. Ashok Kumar Koti, Professor & Students Welfare Officer   Mobile:   09448461738""" + question
    elif 'courses' in question or 'departments' in question or 'programs' in question or 'branches' in question:
        question = """Presently the Institute has 9 departments which cater to the needs of all UG and PG courses to take care of all academic and other activities related to the students, faculty and staff of the departments and the Institute as a whole. The general bird's eye view of departments are as given below.\n1. Civil Engineering (Established: 1982, Courses Offered: B.E. in Civil Engineering, M.Tech in Geotechnical Engineering, Intake: 60+3* for B.E., 18 for M.Tech)\n2. Electronics and Communication Engineering (Established: 1982, Courses Offered: B.E. in Electronics and Communication Engineering, Intake: 90+5*)\n3. Computer Science and Engineering (Established: 1984, Courses Offered: B.E. in Computer Science and Engineering, M.Tech in Computer Science and Engineering, Intake: 120+6* for B.E., 18 for M.Tech)\n4. Mechanical Engineering (Established: 1982, Courses Offered: B.E. in Mechanical Engineering, Intake: 60+3*)\n5. Chemical Engineering (Established: 1994, Courses Offered: B.E. in Chemical Engineering, Intake: 30+2*)\n6. CSE (Artificial Intelligence & Machine Learning) (Established: 2020, Courses Offered: B.E. in CSE (Artificial Intelligence & Machine Learning), Intake: 60+3*)\n7. CSE (Data Science) (Established: 2022, Courses Offered: B.E. in CSE (Data Science), Intake: 60+3*)\n8. CSE (Cyber Security) (Established: 2023, Courses Offered: B.E. in CSE (Cyber Security), Intake: 60+3*)\n9. Applied Science and Humanities (Established: 1982, Courses Offered: B.E., Intake: N.A)\n10. Master of Computer Application (Established: 1998, Courses Offered: MCA, Intake: 60)""" + question
    elif 'board of directors' in question or 'bod' in question or 'board of governors' in question or 'bog' in question or 'governing body' in question or 'members of' in question:
        question =""" Board of Governors : \n1 Er. Eshwar B. Khandre - Chairperson \n2 Sri Keshavrao Nitturkar - Member \n3 Er. Amarkumar B. Khandre - Member, Industrialist \n4 Dr. G. Ravindranath - Member \n5 Dr. R. Sakthivel - Member \n6 Prof. H. U. Talawar - Member \n7 Dr. K. V. Jaykumar - Member, Educationist \n8 Dr. S. C. Pilli - Member, Educationist \n9 Dr. h. G. Shekharappa - Member, Educationist \n10 Dr. Shivakumar andure - Member \n11 Dhiraj Deshpande - Member \n12 Dr.Udaykumar kalyane - Member-Secretary""" + question
    elif 'hod' in question or 'head of department' in question :
        question = """Head of Departments at BKIT :\
            1\tDr. Prashant Sangulagi\tAssociate Professor & HOD\tElectronics and Communication\tM.Tech, Ph.D\t12\n
            2\tDr. Sangamesh Jayprakash Kalyane\tProfessor And HOD\tCSE\tM.Tech, Ph.D\t14\n
            3\tBasawaraj Manikrao Patil\tAssociate Professor & HOD\tME, (Ph.D)\tMechanical Engineering\t28\n
            4\tDr. Mallappa Annarao Devani\tProfessor & HOD\tM.Tech, Ph.D\tChemical Engineering\t25\n
            5\tDr. Ashok Kumar Koti\tProfessor & HOD\tM.Sc, Ph.D\tMathematics\t31\n
            6\tDr. Sangshetty Kalyane\tProfessor & HOD\tM. Sc, Ph.D\tPhysics\t29\n
            7\tDr. D.Vijaykumar Durg\tProfessor & HOD\tM.Sc, Ph.D\tChemistry\t31\n
            8\tYogesh V. Gundge\tAssistant Professor & HOD\t\tM.C.A.\tMCA, (Ph.D)\t16\n
            9\tMallikarjun D. Honna\tAssociate Professor & HOD\tM.Tech\t28\tCivil engineering\n
            """
    elif 'admission' in question or 'procedure' in question or 'requirements' in question or 'eligibility' in question or 'criteria' in question:
        question = """Admission Procedure \nInterested students seeking admission must check their eligibility criteria and fee structure for the desired course.
            Confirm availability of seats of respective branch from Admission Section Superintendent Shri Vijaykumar B. or the Principal on Phone +91-9448459916\nInterested candidate after confirming seat availability should download application form.\nFill up the Application form as well as undertaking form neatly(attached with application form).\nPaste one colored passport size photograph on the application form and enclose 06 nos. of colored passport size and 06 nos. colored stamp size photographs.\nAlso submit original certificates as well as 04 sets of photocopies of\n10th Marks Card/Certificate having date of birth.\nIntermediate/12th/PUC marks card.\nDegree marks card \nTransfer Certificate.\nMigration Certificate\nProof of writing Entrance Test.\nAlong with application, submit D.D. for the whole amount as per fee structure or at least initially for Rs. 40,000/- in favor of "Principal, Bheemanna Khandre Institute of Technology, Bhalki " Payable at Bhalki either for State Bank of India , Bhalki(Code No.5534 ) or State Bank of Hyderabad, Bhalki(Code No.0241) or Canara Bank, Bhalki (Code No.869 ).\nOnly after receipt of full fees, the seat will be confirmed by The Institute.\nAllotment of Course and confirmation of Seat will be made only to those candidates who have paid the full fees and in the order of "First Come First Serve" basis.\nWithout payment of full fees, seat will not be confirmed to any one and risk will lie solely with the candidate regarding confirmation of seats and allotment of course.\nAfter filling in the application form with undertaking complete in all respects, please send the application form along with all enclosures ( Certificates, Photo, D.D etc) by speed post/registered post to " The Principal, Bheemanna Khandre Institute of Technology, Bhalki-585328. Dist.- Bidar (Karnataka State)".\nPlease ensure from the checklist about enclosures to be sent along with filled in application form.\nFilled in Application form.\nUndertaking.\ncolored Photograph(6 copies. passport size & 6 nos. stamp size).\nDemand Draft.\nOriginal certificates plus 4 sets of photocopies copies of.\n10th marks card Certificate containing date of birth.\n12th/Intermediate/PUC Marks card\nDegree marks card.\nTransfer Certificate.\nMigration \nCertificate.\nProof of writing any entrance test for engineering admission as mentioned in eligibility criteria.\nRegarding hostel, admission to the hostel will be given at the time of reporting for attending the classes on first come first serve basis. Before seeking admission to the hostel one must submit D.D. for the requisite amount as mentioned in the fee structure for the hostel and mess.\nOpening date of the college for the first year will be intimated by the admission section of the Institute\nFor any further details/enquiry/clarification etc. please contact Shri Vijaykumar B. on 09448459916.\n\n\nEligibility  U.G (B.E All Branches), P.G (M.Tech, M.C.A)\n\nU.G (B.E All Branches)\nAdmission Under Management Quota\nProof of writing of entrance test of COMED-K or AIEEE/JEE(Main) or any State Level Test or National Level Test.\nPass in PUC/Intermediate/ 10+2 with minimum 45 % marks in the group of Physics and Mathematics as compulsory subjects along with Chemistry /Bio-Technology/ Biology/Computer / Electronics as well as pass in English. In case of SC/ST/OBC candidate, it should not be less than 40%.\nAdmission Under Government Quota\nA Candidate must belong to Karnataka as per eligibility criteria prescribed by Government of Karnataka.\nPass in PUC/Intermediate/ 10+2 with minimum 45 % marks in the group of Physics and Mathematics as compulsory subjects along with Chemistry /Bio-Technology/ Biology/Computer / Electronics as well as pass in English.\nBased on the ranks obtained in CET-Karnataka.\nP.G (M.Tech.)\nPass in B.E/ B.Tech of VTU or any other university/institution or any other examination recognized as equivalent examination with 50% aggregate marks in related branch/discipline. Candidate should have valid GATE score or PGCET qualified rank.\nP.G (M.C.A)\nWriting of entrance test of PGCET of Karnataka or K-MAT or any National level test is compulsory.\nGraduate from any University recognized by VTU, Belgaum having studied Mathematics / Statistics / Computer Science / Computer Programming / Computer Applications as one of the subjects either at 10+2 level or at Degree level with 50% and in case of SC/ST candidate, it shall not less than 45% marks.
        """ + question
    else:
      if 'electronics' in question or ' ec ' in question.lower() or ' ece ' in question.lower():
        return  """ Faculty of Electronics and Communcation Engineering at BKIT\n
           Sl No.\tName\tDesignation\tDepartment\tQualification\tExperience in Years\n
            1\tDr. Udaykumar Kalyane\tProfessor & Principal\tElectronics and Communication\tM.Tech, Ph.D\t21\n
            2\tDr. Prashant Sangulagi\tAssociate Professor & HOD\tElectronics and Communication\tM.Tech, Ph.D\t12\n
            3\tDr. Bardabadi Suryakanth\tProfessor\tElectronics and Communication\tM.Tech, Ph.D\t28\n
            4\tDr. Kalpana Chikatwar\tProfessor\tElectronics and Communication\tME, Ph.D\t24\n
            5\tDr. Manjula S\tProfessor\tElectronics and Communication\tM.Tech, Ph.D\t21\n
            6\tDr. Ramesh Dhavalegar\tAssociate Professor\tElectronics and Communication\tM.Tech, Ph.D\t25\n
            7\tSharanabasappa Belamgi\tAssociate Professor\tElectronics and Communication\tM.Tech, (Ph.D)\t25\n
            8\tSatish Kannale\tAssociate Professor\tElectronics and Communication\tM.Tech, (Ph.D)\t10\n
            9\tSanjeev Gogga\tAssociate Professor\tElectronics and Communication\tM.Tech, (Ph.D)\t24\n
            10\tDhiraj Deshpande\tAssociate Professor\tElectronics and Communication\tM.Tech\t28\n
            11\tVijay Katagi\tAssistant Professor\tElectronics and Communication\tM.Tech\t13\n
            12\tPrahallad S\tAssistant Professor\tElectronics and Communication\tM.Tech\t13\n
            13\tHameed Miyan\tAssistant Professor\tElectronics and Communication\tM.Tech, (Ph.D)\t14\n
            14\tSantosh Ammanna\tAssistant Professor\tElectronics and Communication\tM.Tech\t06\n
            15\tPrashant Khandale\tAssistant Professor\tElectronics and Communication\tM.Tech\t06\n
            16\tPooja Patne\tAssistant Professor\tElectronics and Communication\tM.Tech\t05\n
            17\tShilpa Patne\tAssistant Professor\tElectronics and Communication\tM.Tech\t05\n
            18\tRaveena Goure\tAssistant Professor\tElectronics and Communication\tM.Tech\t03\n
            19\tShivaleela Veershetty\tAssistant Professor\tElectronics and Communication\tM.Tech\t02\n
            20\tRajkumar M. Vadgave\tAssistant Professor\tElectronics and Communication\tM.Tech\t04\n
            21\tSavitri Kanshette\tAssistant Professor\tElectronics and Communication\tM.Tech\t02\
            22\tJatla Chandra Shekhar\tAssistant Professor\tElectronics and Communication\tM.Tech\t01\n
            23\tChannabasamma Pasarge\tAssistant Professor\tElectronics and Communication\tM.Tech\t01\n
            24\tSangeeta Hiremath\tAssistant Professor\tElectronics and Communication\tM.Tech\t01\n
            25\tYouvan Shivappa\tAssistant Professor\tElectronics and Communication\tM.Tech\t01"""
          
      elif 'computer' in question.lower() or ' cs ' in question or ' cse ' in question.lower():
          question = """ Faculties of Computer Science and Engineering : \n 
            Sl No.\tName\tDesignation\tDepartment\tQualification\tExperience in Years\n
              1\tHanmanth Shankarrao Kulkarni\tAssociate Professor \tCSE\tM.Tech\t28\n
              2\tDr. Mallikarjun Basavaraj Mugli\tProfessor\tCSE\tM.Tech, Ph.D\t28\n
              3\tDr. Basavaraj Shantkumar Prabha\tProfessor\tCSE\tM.Tech, Ph.D\t23\n
              4\tDr. Sangamesh Jayprakash Kalyane\tProfessor And HOD\tCSE\tM.Tech, Ph.D\t14\n
              5\tGeeta Patil\tAssistant Professor\tCSE\tM.Tech, (Ph.D)\t18\n
              6\tMd. Rafeeq\tAssistant Professor\tCSE\tM.Tech, (Ph.D)\t23\n
              7\tMore Vishal Ramrao\tAssistant Professor\tCSE\tM.Tech, (Ph.D)\t15\n
              8\tShilpa Shivdas Harnale\tAssistant Professor\tCSE\tM.Tech, (Ph.D)\t16\n
              9\tAmbika Basavaraj Mangalgi\tAssistant Professor\tCSE\tM.Tech, (Ph.D)\t13\n
              10\tSudarshan Gurunath Adeppa\tAssistant Professor\tCSE\tM.Tech\t11\n
              11\tDr. Chamundeshwari Kalyane\tAssistant Professor\tCSE\tM.Tech, Ph.D\t11\n
              12\tJagadanna Bandeppa\tAssistant Professor\tCSE\tM.Tech\t07\n
              13\tPriyanka Patil\tAssistant Professor\tCSE\tM.Tech\t08\n
              14\tUzma Khanam Younus\tAssistant Professor\tCSE\tM.Tech\t06\n
              15\tVijaylaxmi Tippanna\tAssistant Professor\tCSE\tM.Tech\t06\n
              16\tAshwini Basavaraj Bhangure\tAssistant Professor\tCSE\tM.Tech\t05\n
              17\tAshavini Suryakanth Satbige\tAssistant Professor\tCSE\tM.Tech\t05\n
              18\tBasavaraj Shrishail\tAssistant Professor\tCSE\tM.Tech\t01\n
              19\tAjaykumar Narsappa\tAssistant Professor\tCSE\tM.Tech\t01\n
              20\tVachanashree Patne\tAssistant Professor\tCSE\tM.Tech\t01\n
              21\tAshwini Patil\tAssistant Professor\tCSE\tM.Tech\t01\n
              22\tVarsharani\tAssistant Professor\tCSE\tM.Tech\t01\n
              23\tAbhishek Shivakumar Sali\tAssistant Professor\tCSE\tM.Tech\t01\n
              Faculty of CSE (Artificial Intelligence & Machine Learning) : \n
              Sl No.\tName\tDesignation\tDepartment\tQualification\tExperience in Years\n
              1\tDr. Shivkumar Andure\tProfessor, HOD\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech, Ph.D\t21\n
              2\tDhanraj Biradar\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t12\n
              3\tShruti\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t03\n
              4\tReba Rani\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t01\n
              5\tMeghana Mahindrakar\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t02\n
              6\tMamta\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t02\n
              7\tKeertirani\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t02\n
              8\tArati Biradar\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t02\n
              9\tArati Bachhanna\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t06\n
              10\tVijeta Dhuleholi\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t01\n
              11\tSupriya Biradar\tAssistant Professor\tCSE (Artificial Intelligence & Machine Learning)\tM.Tech\t01\n
              Faculty of CSE (Data Science) : \n
              Sl No.\tName\tDesignation\tDepartment\tQualification\tExperience in Years\n
              1\tAkshata Patil\tAssistant Professor\tCSE (Data Science)\tM.Tech\t02\n
              2\tAshwini Pandit Reddy\tAssistant Professor\tCSE (Data Science)\tM.Tech\t02\n
              3\tNikita Laxmanrao Biradar\tAssistant Professor\tCSE (Data Science)\tM.Tech\t02\n""" + question
      elif 'mechanical' in question.lower() :
          question = """FACULTY of Mechenical Engineering : 
          Sl No.\tName\tDesignation\tQualification\tDepartment\tExperience in Years\n
          1\tBasawaraj Manikrao Patil\tAssociate Professor & HOD\tME, (Ph.D)\tMechanical Engineering\t28\n
          2\tDr. Basawaraj Kawdi\tProfessor\tME, Ph.D\tMechanical Engineering\t32\n
          3\tShivkumar Biradar\tAssociate Professor\tME\tMechanical Engineering\t28\n
          4\tJairaj Siddan\tAssociate Professor\tM.Tech\tMechanical Engineering\t27\n
          5\tUmakanth Mathapati\tAssistant Professor\tM.Tech,(Ph.D)\tMechanical Engineering\t15\n
          6\tJeevaraj Samuel\tAssistant Professor\tM.Tech\tMechanical Engineering\t14\n
          7\tAnandkumar Telang\tAssistant Professor\tM.Tech,(Ph.D)\tMechanical Engineering\t15\n
          8\tShrinath R. Bhosle\tAssistant Professor\tM.Tech,(Ph.D)\tMechanical Engineering\t14\n
          9\tRaj Reddy\tAssistant Professor\tM.Tech,(Ph.D)\tMechanical Engineering\t11\n
          10\tEbinezar Jayker Devkote\tAssistant Professor\tM.Tech\tMechanical Engineering\t10\n
          11\tAdarsh Adeppa\tAssistant Professor\tM.Tech,(Ph.D)\tMechanical Engineering\t11\n
          12\tAmarkumar U. Vatambe\tAssistant Professor\tM.Tech, (Ph.D)\tMechanical Engineering\t07\n
          13\tSrikant Tilekar\tAssistant Professor\tM.Tech\tMechanical Engineering\t03""" + question
      elif 'chemical ' in question or ' che ' in question:
          question = """ Faculty of Chemical Engineering : \n
          Sl No. Name\tDesignation\tQualification\tDepartment\tExperience in Years\n
          1 Dr. Mallappa Annarao Devani\tProfessor & HOD\tM.Tech, Ph.D\tChemical Engineering\t25\n
          2 Babu Satbaji Patil\tAssociate Professor\tM.Tech\tChemical Engineering\t26\n
          3 Dr. Akilesh Prabhakar Khapre\tAssociate Professor\tM.Tech, Ph.D\tChemical Engineering\t05\n
          4 Sambhaji Ramrao Birajdar\tAssistant Professor\tM.Tech\tChemical Engineering\t02""" + question
      elif 'mathematics' in question:
          question = """ Faculty of Mathematics : \n
          Sl No.\tName\tDesignation\tQualification\tDepartment\tExperience in Years\n
          1\tDr. Ashok Kumar Koti\tProfessor & HOD\tM.Sc, Ph.D\tMathematics\t31\n
          2\tDr. Geetanjali Alle\tProfessor\tM.Sc, Ph.D\tMathematics\t24\n
          3\tDr. Dayanand Kallur\tProfessor\tM.Sc. Ph.D\tMathematics\t23\n
          4\tOmkar Chenbasappa Kadale\tAssistant Professor\tM.Sc.\tMathematics\t03\n
          5\tShreenidhi N.\tAssistant Professor\tM.Sc.\tMathematics\t01\n
          6\tAmbika S.\tAssistant Professor\tM.Sc.\tMathematics\t03\n
          7\tBharatrao Biradar\tAssistant Professor\tMA\tMathematics\t03\n
          8\tSangappa Solmal\tAssistant Professor\tMA\tMathematics\t03\n""" + question
      elif 'physics' in question:
          question = """Faculty of Physiscs : \n
            Sl No.\tName\tDegination\tQualification\tDepartment\tExperience in Years\n
            1\tDr. Sangshetty Kalyane\tProfessor & HOD\tM. Sc, Ph.D\tPhysics\t29\n
            2\tNagraj C. Patil\tAssistant Professor\tM.Sc\tPhysics\t25\n
            3\tPrashant Vijaykumar Mule\tAssistant Professor\tM.Sc\tPhysics\t04\n
            4\tAnilkumar Punna\tAssistant Professor\tMA\tPhysics\t02\n
            5\tDr. Kapil Reddy\tAssistant Professor\tBAMS\tPhysics\t02\n
            6\tKavita Balte\tAssistant Professor\tMA\tPhysics\t02\n""" + question
      elif 'chemistry ' in question:
          question = """Faculty of Chemistry : \n
          Sl No. Name\tDesignation\tQualification\tDepartment\tExperience in Years\n
          1\tDr. D.Vijaykumar Durg\tProfessor & HOD\tM.Sc, Ph.D\tChemistry\t31\n
          2\tDr. Anilkumar Kogde\tProfessor\tM.Sc, Ph.D\tChemistry\t21\n
          3\tMd. Nadeem Miyan\tAssistant Professor\tM.Sc, (Ph.D)\tChemistry\t10\n
          4\tSushma Swamy\tAssistant Professor\tM.Sc\tChemistry\t06\n
          5\tAshok Bhandary\tAssistant Professor\tMA\tChemistry\t08""" + question
      elif ' mca ' in question or 'm.c.a' in question:
          question = """Faculty of M.C.A. are : \n 
          Sl No.\tName\tDesignation\tDepartment\tQualification\tExperience in Years\n
          1\tYogesh V. Gundge\tAssistant Professor & HOD\t\tM.C.A.\tMCA, (Ph.D)\t16\n
          2\tSunilkumar Sangme\tAssociate Professor\t\tM.C.A.\tMCA, M.Phil\t25\n
          3\tGayatri Muglie\tAssistant Professor\t\tM.C.A.\tMCA, (Ph.D)\t14\n
          4\tPoojarani Bone\tAssistant Professor\t\tM.C.A.\tM.Tech\t06\n
          5\tKaveri Reddy\tAssistant Professor\t\tM.C.A.\tM.Tech\t06\n
          6\tNeelambika D.\tAssistant Professor\t\tM.C.A.\tMCA\t06\n
          7\tBhairewar Sandhya Gopalrao\tAssistant Professor\tM.C.A.\tMCA\t01\n
          8\tPriyanka Kamshetty\tAssistant Professor\tM.C.A.\tMCA\t01\n
          9\tRekha Bhair\tAssistant Professor\t\tM.C.A.\tMCA\t01\n""" + question
      elif 'civil ' in question or ' ce ' in question:
          question = """FACULTY of Civil Engineering :\n 
          Sl No.\tName\tDesignation\tQualification\tExperience in Years\tDepartment\n
          1\tMallikarjun D. Honna\tAssociate Professor & HOD\tM.Tech\t28\tCivil engineering\n
          2\tDr. Ashok Gandagi\tProfessor\tME, Ph.D\t27\tCivil engineering\n
          3\tDr. Giridhari Tiwari\tProfessor\tM.Tech, MBA, Ph.D\t27\tCivil engineering\n
          4\tDr. Rajshekhar Mathapati\tAssociate Professor\tME, Ph.D\t26\tCivil engineering\n
          5\tSantosh Dhadde\tAssistant Professor\tM.Tech, (Ph.D)\t11\tCivil engineering\n
          6\tMd. Khaja Moniuddin\tAssistant Professor\tM.Tech, (Ph.D)\t14\tCivil engineering\n
          7\tSurekha Shaka\tAssistant Professor\tM.Tech\t09\tCivil engineering\n
          8\tGoudappa Biradar\tAssistant Professor\tM.Tech\t10\tCivil engineering\n
          9\tNagesh Mustapure\tAssistant Professor\tM.Tech\t08\tCivil engineering\n
          10\tVinodkumar A. Gama\tAssistant Professor\tM.Tech\t11\tCivil engineering\n
          11\tUllas Jaje\tAssistant Professor\tM.Tech\t09\tCivil engineering\n
          12\tVishal Manohar\tAssistant Professor\tM.Tech\t05\tCivil engineering\n
          13\tNeha Meti\tAssistant Professor\tM.Tech\t04\tCivil engineering\n
          14\tBalaji K. Deshmukh\tAssistant Professor\tM.Tech\t07\tCivil engineering\n
          15\tPoojashri Mulge\tAssistant Professor\tM.Tech\t03\tCivil engineering""" + question
    return after_rag_chain.invoke(question)

 

import streamlit as st
from dotenv import load_dotenv
import os
import shelve

load_dotenv()

st.title("Campus Chatter")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "meta-llama/Llama-2-70b-chat-hf"


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", []) 


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        for response in process_input(prompt):
            full_response += response or ""
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)

