from setuptools import setup,find_packages
from typing import List
HYPEN_E_DOT="-e."



def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        return requirements

        if HYPEN_E_DOT in requirment:
            requirment.remove("HYPEN_E_DOT")




setup(
    name='SIGN_LANGUAGE_DETECTION',
    version="0.0.1",
    author="Abhishek Upadhyay",
    author_email="abhishekupadhyay9336@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
    
    
    
)