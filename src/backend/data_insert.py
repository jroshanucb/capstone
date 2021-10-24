# INSERT INTO table_name (column1, column2, column3, ...)
# VALUES (value1, value2, value3, ...);

# insert into  public.event_images 
# (event_id, image_group_id, image_url_1, image_url_2, image_url_3, species_name, count, load_date, blank_image)
# values (1, 'SSWI000000000205203', 'https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/SSWI000000000205203A.jpg', 
# 		'https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/SSWI000000000205203B.jpg', 
# 		'https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/SSWI000000000205203C.jpg',
# 		'deer', 1, to_date('24-10-2021','DD-MM-YYYY'), false)

from db_conn import load_db_table
from db_conn import config
import pandas as pd
import psycopg2

imagesDict = {}

# Sample values in the CSV file
# format: CLASS_SPECIES, CLASS_KEY, KEY_COUNT, KEY_TEXT, TRIGGER_ID, CAMERA_SEQ_NO, TRIGGER_DATETIME
# "Skunk, Striped",SKUNKSTRIPED_AMT,1,NA,SSWI000000016160618,28853,15-Dec-19
# Deer,DEER_ADULT_ANTLERLESS_AMT,1,NA,SSWI000000020299896,32164,16-Apr-19
# Wolf,WOLF_AMT,2,NA,SSWI000000016482792,61758,7-Feb-20
# Squirrels and Chipmunks,SQUIRREL_AMT,1,NA,SSWI000000017494673,54241,14-Jun-20
# Elk,ELK_ADULT_ANTLERLESS_AMT,1,NA,SSWI000000015519932,63902,28-Oct-19
# "Grouse, Ruffed",GROUSE_AMT,1,NA,SSWI000000014892341,55621,21-Sep-19
# Elk,ELK_ADULT_ANTLERLESS_AMT,1,NA,SSWI000000015736964,28204,12-Nov-19
# "Skunk, Striped",SKUNKSTRIPED_AMT,1,NA,SSWI000000020971116,58253,1-Mar-21
# Wolf,WOLF_AMT,1,NA,SSWI000000016300839,54977,10-Jan-20
# "Skunk, Striped",SKUNKSTRIPED_AMT,1,NA,SSWI000000003780629,23378,9-Jul-16
# Elk,ELK_COLLAR_PRESENT,NA,Y,SSWI000000018039845,63902,14-Jun-20
# Wolf,WOLF_AMT,2,NA,SSWI000000018270062,61758,5-Jul-20
# Elk,ELK_ADULT_UNKNOWN_AMT,1,NA,SSWI000000015060827,63902,13-Sep-19
# Elk,ELK_ADULT_ANTLERLESS_AMT,4,NA,SSWI000000015060749,63902,4-Sep-19
# "Skunk, Striped",SKUNKSTRIPED_AMT,1,NA,SSWI000000017503640,33111,20-Apr-20
# "Skunk, Striped",SKUNKSTRIPED_AMT,1,NA,SSWI000000020586032,81364,26-Dec-20

with open('Berkeley-sswi_metadata.csv') as d:
    line = d.readline() # ignore first line which has column headings
    line = d.readline() # read next line
    while line:
        values = line.split(',')
        key = values[4].strip()
        # dictionary values has format: species_name, count
        if key in imagesDict:
            print("Key found in dictionary: ", key)
        species_count = values[2].strip()
        if species_count == "NA":
            species_count = "0"
        imagesDict[key] = values[0].strip() + ", " + species_count
        line = d.readline() # read next line
d.close()

config_db = "database.ini"
params = config(config_db)
conn = psycopg2.connect(**params)
print("postgres database connection successful")

with open('filenames.txt') as f:
    event_id = 0
    line1 = f.readline()
    if line1:
        line2 = f.readline()
        if line2:
            line3 = f.readline()
        while line3:
        # if line3:
            print(line1, line2, line3)
            event_id += 1
            name1 = line1.strip().split('.')
            name2 = line2.strip().split('.')
            name3 = line3.strip().split('.')
            group_id = name1[0][:-1]

            # add assertion that the 3 image group ids are same
            url1 = "https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/" + name1[0] + "." + name1[1]
            url2 = "https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/" + name2[0] + "." + name2[1]
            url3 = "https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/" + name3[0] + "." + name3[1]

            # values = imagesDict[group_id, "empty"]
            if group_id in imagesDict:
                blank_image = "false"
                values = imagesDict[group_id]
                print(values)
                species_name_count = values.split(',')
                species_name = species_name_count[0]
                count = species_name_count[1]
            else:
                blank_image = "true"
                species_name = ""
                count = "0"

            load_date = "to_date('24-10-2021','DD-MM-YYYY')"

            sql_stmt = "insert into public.event_images (event_id, image_group_id, image_url_1, image_url_2, image_url_3, species_name, count, blank_image, load_date) "
            sql_stmt += "values (" + str(event_id) + ", '" + group_id + "', '" + url1 + "', '" + url2 + "', '" + url3 + "', '" + species_name + "', " 
            sql_stmt += count + ", " + blank_image + ", " + load_date + ")"
            print(sql_stmt)

            cur = conn.cursor()
            cur.execute(sql_stmt)
            conn.commit()

            # print("print from the DB query run: ", df)

            line1 = f.readline()
            if line1:
                line2 = f.readline()
                if line2:
                    line3 = f.readline()
f.close()