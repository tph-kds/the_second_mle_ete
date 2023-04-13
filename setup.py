from setuptools import find_packages , setup
from typing import List

HYPEN_E_DOT = '-e.'
def get_requirement(file_path:str)->List[str]:
    # '''
    #     This function will return this list of requirements
    # '''

    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n" , "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name = 'thesencondmleete' , 
    version = '0.0.1' , 
    author = 'Hero' , 
    author_email = "hungcompo123@gmail.com" , 
    packages =  find_packages() , 
    install_requires = get_requirement("D://ML Engineer Project/Second Project MLE on YTB/requirements.txt")
)