import os, imghdr

directories = ["C:\\Users\\HP\\Desktop\\Science Fair Project\\Image Data\\NormalEEG",
			   "C:\\Users\\HP\\Desktop\\Science Fair Project\\Image Data\\AbnormalEEG"]

def main():
	for directory in directories:
		for i, file in enumerate(os.listdir(directory)):
			extension = imghdr.what(os.path.join(directory,file))
			res = (os.path.join(directory,file), f"{directory}\\{i}.{extension}")
			os.rename(res[0] , res[1])
			print(res[1])

if __name__ == "__main__":
	main()

