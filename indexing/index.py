import sys
import os

class HashTable:
	#hashlist
	table_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]] #20 cells
	length = len(table_list)

	def get_to_hash_table(self,value,path_to_the_file):
		#get the mod position
		pos = value % self.length
		#append to the innerlist
		self.table_list[pos].append((value,path_to_the_file))

	def search_element(self,file_to_search):
		#get the position of the cell
		pos = int(file_to_search) % self.length 
		#print(pos) 
		cell = self.table_list[pos]

		for name,loc in cell:

			if name == int(file_to_search):
				status = loc
				break
			else:
				status = "Oops..!not found"

		return pos,status

"""def write_to_ext_file(hashtable):
	total_count = 0
	with open("index.txt","a") as fp:
		for count in range(len(hashtable)):
			for name,loc in hashtable[count]:
				fp.write(str(name) + ",")
			fp.write("\n")"""
				

if __name__ == '__main__':

	hash_table = HashTable()
	path_file = "/home/madhi/Documents/python programs/neuralnetworks/fp/Reuters21578-Apte-115Cat/training"
	folders_in_path_file = os.listdir(path_file)
	#files in the inner folder
	for folders in folders_in_path_file:
		#set path inside the folder
		inner_folder_path = path_file + "/" + folders
		getfiles = os.listdir(inner_folder_path)
		#append to the list
		for org_files in getfiles:
			# original_file name and path_of_the file is passed as a parameter
			hash_table.get_to_hash_table(int(org_files),inner_folder_path+"/"+org_files) #send the file name as integer to calculate mod

	
	#write_to_ext_file(hash_table.table_list) #save to the ext file

	user_input = input("Enter the file to search:")
	pos,status = hash_table.search_element(user_input)
	print("File {0} located at cell: {1} \n location: {2}".format(user_input,pos,status))
	