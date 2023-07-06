import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
import logging

# 創建一個logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # 設定最低日誌等級為DEBUG，這樣所有等級的日誌都能被捕捉到

# 創建一個處理器，寫入INFO及以上等級的日誌到info.log檔案
info_handler = logging.FileHandler('auto.log')
info_handler.setLevel(logging.INFO)

# 再創建一個處理器，寫入ERROR及以上等級的日誌到error.log檔案
error_handler = logging.FileHandler('error.log')
error_handler.setLevel(logging.ERROR)

# 創建一個日誌格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 將該格式器添加到兩個處理器
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# 將兩個處理器添加到我們的logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)



def login():
    chrome = webdriver.Chrome()
    chrome.get("https://0800056476.sme.gov.tw/smeloans/tele_credit/login.php")
    
    cookie_name = 'PHPSESSID'  # 將 'some_cookie' 替換成你想要獲取的cookie的名稱
    cookie = chrome.get_cookie(cookie_name)
    cookie = cookie['value']

    
    # 定位到img標籤
    img_tag = chrome.find_element(By.XPATH, '//img[contains(@src, "checkCode.php")]')
    # 獲得src屬性的值
    img_src = img_tag.get_attribute('src')
    # 從src中抽取你需要的數字
    num = img_src.split('=')[1]
    

    # 找到帳號輸入框並輸入帳號
    username = chrome.find_element(By.XPATH, '//*[@id="inputEmail3"]')

    username.clear()
    username.send_keys('teleadmin')  # 將 'XXX' 替換成你的帳號

    # 找到密碼輸入框並輸入密碼
    password = chrome.find_element(By.XPATH, '//*[@id="kppaa"]')
    password.clear()
    password.send_keys('teleadmin!QAZ!QAZ2wsx2wsx')  # 將 'BBB' 替換成你的密碼

    # 找到驗證碼輸入框並輸入驗證碼
    checkcode = chrome.find_element(By.XPATH, '//*[@id="code"]')
    checkcode.clear()
    checkcode.send_keys(num)


    # 點擊登入按鈕
    login_button = chrome.find_element(By.XPATH,'//button[normalize-space()="登入"]')
    login_button.click()
#     chrome.close()
    print("登入成功")
    logger.info("登入成功")
    return cookie



def snd_line(message):
    
    headers = {
        "Authorization": "Bearer " + "3nybyrnsybWvlg7G7h18A0dyJgXFytlVpr2GRXY0evc",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    params = {"message": message}

    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, params=params)
    print(r.status_code)  #200

    
    

cookie = login()

main_url = 'https://0800056476.sme.gov.tw/smeloans/tele_credit/'
url = 'https://0800056476.sme.gov.tw/smeloans/tele_credit/tele_credit_a2_list.php'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6',
    'Cache-Control': 'max-age=0',
    'Sec-Ch-Ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}
cookies = {
    'PHPSESSID': cookie,
}



try:
    response = requests.get(url, headers=headers, cookies=cookies)
except Exception as e:
    print(e)
    logger.info(e)

    snd_line(e)


# print(response.status_code)
# print(response.text)

soup = BeautifulSoup(response.text, 'html.parser')

# Extract all "a" tags with "href" attribute
links = [a.get('href') for a in soup.find_all('a', href=True) if '_umemberp' in a.get('href')]


## 驗證是否需要重新登入
try:
    scripts = soup.find_all('script')
    for script in scripts:
        if '無觀看權限!!' in script.string:
            print(script.string)
            snd_line("連線中斷，請重新登入\nhttps://0800056476.sme.gov.tw/smeloans/tele_credit/login.php")
except Exception as e:
    print(e)
    logger.info(e)

        
        
# 找到所有的<a>標籤
tmp_links = soup.find_all('a')
send_param = ""
# 透過正則表達式分析每個href屬性
for link in tmp_links:
    href = link.get('href')
    if href:
        match = re.search(r'send=(.*?)&', href)
        if match:
            send_param = match.group(1)
            print(send_param)
            logger.info(send_param)

            break
            
link_df = pd.DataFrame()
link_df['link'] = links
link_df['case_id'] = link_df['link'].apply(lambda x: x.split("/fs")[1].split("_")[0] if "/fs" in x and "_" in x else None)
# link_df['ent_or_rep'] = link_df['link'].apply(lambda x: x.split(".pdf")[0].split("-")[1] if "-" in x and ".pdf" in x else None)
link_df['ent_or_rep'] = link_df['link'].apply(lambda x: x.split(".pdf")[0].split("-")[1] if "-" in x and ".pdf" in x 
                                              else x.split(".jpeg")[0].split("-")[1] if "-" in x and ".jpeg" in x 
                                              else x.split(".jpg")[0].split("-")[1] if "-" in x and ".jpg" in x 
                                              else x.split(".png")[0].split("-")[1] if "-" in x and ".png" in x 
                                              else x.split(".tif")[0].split("-")[1] if "-" in x and ".tif" in x 
                                              else x.split(".tiff")[0].split("-")[1] if "-" in x and ".tiff" in x 
                                              else None)
link_df['file'] = link_df['link'].apply(lambda x: x.split("/")[1] if "/" in x  else None)



# df = pd.read_html(response.text)
df = pd.read_html(response.text)
dfs = df[0]
dfs['銀行代碼'] = dfs['銀行代碼'].astype(str).str.zfill(3)
dfs['分行代碼'] = dfs['分行代碼'].astype(str).str.zfill(4)
dfs


# 將dataframe依據 'case_id' 分組，並對 'ent_or_rep' 欄位進行 unique 操作
unique_ent_or_rep = link_df.groupby('case_id')['ent_or_rep'].unique()

# 過濾出 'ent_or_rep' 同時包含 1 和 2 的 'case_id'
filtered_cases = unique_ent_or_rep[unique_ent_or_rep.apply(lambda x: set(x) == {'1', '2'})]

# 取出符合條件的 'case_id'
case_ids = filtered_cases.index.tolist()

two_file_case_list = link_df[link_df['case_id'].isin(case_ids)]




def download_file(case_id,file_url,file_name):
    
    
    # 資料夾名稱
    folder_name = case_id
    folder_name = 'datasets\\temp\\'+folder_name
    # 檢查資料夾是否存在
    if not os.path.exists(folder_name):
        # 創建資料夾
        os.makedirs(folder_name)
        
        
    main_url = 'https://0800056476.sme.gov.tw/smeloans/tele_credit/'
    download_url = main_url + file_url

    response = requests.get(download_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in write-binary mode and write the response content to it
        with open(os.path.join(folder_name, file_name), 'wb') as f:
            print(f"Success to download: {download_url}")
            logger.info(f"Success to download: {download_url}")

            f.write(response.content)
    else:
        print(f"Failed to download file :{download_url}")
        logger.info(f"Failed to download file :{download_url}")

#         snd_line(f"Failed to download file :{download_url}")
        
        
shutil.rmtree(r'datasets/temp')

print("下載檔案..")

two_file_case_list.apply(lambda row: download_file(row['case_id'], row['link'], row['file']), axis=1)



print("跑AI辨識..")

from final import agree_ai_check
import pandas as pd
import os
from datetime import datetime, timedelta
import sys
from final import agree_ai_check
import pandas as pd
import os
from datetime import datetime, timedelta
import sys
import time
import logging
import shutil
from PyPDF2 import PdfReader



# 設定 log 檔
# logging.basicConfig(filename='error.log', level=logging.ERROR)
# 獲取當前時間
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 從外部傳入資料夾路徑
# folder_path = sys.argv[1]  
folder_path = r'datasets\temp'

# 檢查資料夾是否存在，若不存在則創建資料夾
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 檢查 folder_path 是不是一個資料夾
if not os.path.isdir(folder_path):
    print("指定的資料夾不存在或不是一個資料夾")
    sys.exit()

# 檢查 folder_path 底下是否是空的
if len(os.listdir(folder_path)) == 0:
    print("指定的資料夾底下是空的")
    sys.exit()





# 列出本機路徑下的資料夾
folder_list = os.listdir(folder_path)



df = pd.DataFrame(columns=['folder', 'file', 'result', 'timestamp'])
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')




from tqdm import tqdm




all_file_list = []
# 列出每個子資料夾下的所有檔案
all_agree_folder_path = 'datasets\\all_agree'
# 檢查資料夾是否存在，若不存在則創建資料夾
if not os.path.exists(all_agree_folder_path):
    os.makedirs(all_agree_folder_path)
    
all_agree_folder_list = os.listdir(all_agree_folder_path)
for subfolder in all_agree_folder_list:
    subfolder_path = os.path.join(all_agree_folder_path, subfolder)
    if os.path.isdir(subfolder_path):  # 確保是一個資料夾，而不是一個檔案
        subfolder_files = os.listdir(subfolder_path)
        print(f"Files in {subfolder}:")
        for file in subfolder_files:
            all_file_list.append(file)

E_agree_P_disagree_folder_path = 'datasets\\E_agree_P_disagree'
# 檢查資料夾是否存在，若不存在則創建資料夾
if not os.path.exists(E_agree_P_disagree_folder_path):
    os.makedirs(E_agree_P_disagree_folder_path)
E_agree_P_disagree_folder_list = os.listdir(E_agree_P_disagree_folder_path)
# 列出每個子資料夾下的所有檔案
for subfolder in E_agree_P_disagree_folder_list:
    subfolder_path = os.path.join(E_agree_P_disagree_folder_path, subfolder)
    if os.path.isdir(subfolder_path):  # 確保是一個資料夾，而不是一個檔案
        subfolder_files = os.listdir(subfolder_path)
        print(f"Files in {subfolder}:")
        for file in subfolder_files:
            all_file_list.append(file)

E_disagree_P_agree_folder_path = 'datasets\\E_disagree_P_agree'
if not os.path.exists(E_disagree_P_agree_folder_path):
    os.makedirs(E_disagree_P_agree_folder_path)
E_disagree_P_agree_folder_list = os.listdir(E_disagree_P_agree_folder_path)
# 列出每個子資料夾下的所有檔案
for subfolder in E_disagree_P_agree_folder_list:
    subfolder_path = os.path.join(E_disagree_P_agree_folder_path, subfolder)
    if os.path.isdir(subfolder_path):  # 確保是一個資料夾，而不是一個檔案
        subfolder_files = os.listdir(subfolder_path)
        print(f"Files in {subfolder}:")
        for file in subfolder_files:
            all_file_list.append(file)


all_disagree_folder_path = 'datasets\\all_disagree'
if not os.path.exists(all_disagree_folder_path):
    os.makedirs(all_disagree_folder_path)
all_disagree_folder_list = os.listdir(all_disagree_folder_path)
# 列出每個子資料夾下的所有檔案
for subfolder in all_disagree_folder_list:
    subfolder_path = os.path.join(all_disagree_folder_path, subfolder)
    if os.path.isdir(subfolder_path):  # 確保是一個資料夾，而不是一個檔案
        subfolder_files = os.listdir(subfolder_path)
        print(f"Files in {subfolder}:")
        for file in subfolder_files:
            all_file_list.append(file)

Fail_folder_path = 'datasets\\Fail'
if not os.path.exists(Fail_folder_path):
    os.makedirs(Fail_folder_path)
Fail_folder_list = os.listdir(Fail_folder_path)
# 列出每個子資料夾下的所有檔案
for subfolder in Fail_folder_list:
    subfolder_path = os.path.join(Fail_folder_path, subfolder)
    if os.path.isdir(subfolder_path):  # 確保是一個資料夾，而不是一個檔案
        subfolder_files = os.listdir(subfolder_path)
        print(f"Files in {subfolder}:")
        for file in subfolder_files:
            all_file_list.append(file)


for folder in tqdm(folder_list):
    print(f"folder : {folder}")
    if folder in all_file_list:
        print(f"已審核過{folder}")
        logger.info(f"已審核過{folder}")
        continue

    folder_full_path = os.path.join(folder_path, folder)  # 資料夾的完整路徑
    files_list = os.listdir(folder_full_path)  # 資料夾內的所有檔案
    if len(files_list) > 0:
        rows = []
        for file in tqdm(files_list, leave=False):  # 給內層的loop也加上進度條，使用leave=False讓內層的進度條在完成後消失
            file_path = os.path.join(folder_full_path, file)  # 檔案的完整路徑
            # 使用 os.path.splitext() 函式來取得檔案的副檔名 & 檔名
            file_extension = os.path.splitext(file_path)[1] 
            file_name = os.path.splitext(file)[0]
            
            try:
                match = re.search(r"_(\d)_(\d)", file_name)
                if match:
                    E_status = int( match.group(1) )
                    P_status = int( match.group(2) )
                else:
                    raise ValueError("無法從檔名中找到授權書同意與否狀態")
            except Exception as e:
                error_msg = f"[{current_time}] 檔案 {file} 檔名錯誤: {e}"
                logging.error(error_msg)
                continue
            # 讀取檔案並執行 agree_ai_check 函式            
            try:
                # pdf檔案內有多頁的情況
                if file_extension == ".pdf":
                    # 讀取 PDF 檔案
                    with open(file_path, "rb") as file_:
                        pdf_reader = PdfReader(file_)
                        page_count = len(pdf_reader.pages)  # 獲取頁數
                    #單頁情況 用檔名判斷用哪個模型
                    if page_count == 1: 
                        try:
                            json_result = agree_ai_check(company_id="123", rep_id="123", location=folder_full_path, agree_file_name=file,E_status=E_status,P_status=P_status)
                        except Exception as e:
                            print(e)
                    #多頁情況 用分類模型判斷用哪個模型
                    elif page_count > 1:
                        try:
                            json_result = agree_ai_check(company_id="123", rep_id="123", location=folder_full_path, agree_file_name=file,E_status=E_status,P_status=P_status) #獲得一個list檔案 裡面是多個json結果
                        except Exception as e:
                            print(e)  
                # 不是PDF檔的情況
                else: 
                    try:
                        json_result = agree_ai_check(company_id="123", rep_id="123", location=folder_full_path, agree_file_name=file,E_status=E_status,P_status=P_status)
                    except Exception as e:
                        print(e)
                if isinstance(json_result, dict):
                    # 提取相關資訊
                    result = json_result.get('message')
                    agree_type = json_result.get('agree_type_final') 
                    # 建立新的資料列
                    row = { 'agree_type':agree_type,'folder': folder, 'file': file, 'result': result, 'timestamp': timestamp}
                    rows.append(row)
                elif isinstance(json_result, list):
                    for json_ in json_result:
                        result = json_.get('message')
                        agree_type = json_.get('agree_type_final') 
                        # 建立新的資料列
                        row = { 'agree_type':agree_type,'folder': folder, 'file': file, 'result': result, 'timestamp': timestamp}
                        rows.append(row)                     
            except Exception as e:
                error_msg = f"[{current_time}] 檔案 {file} 辨識出現錯誤: {e}"
                logging.error(error_msg)
                print(error_msg)
        # 將資料列加入 DataFrame
        df = df.append(rows, ignore_index=True)
    else:
        # 資料夾內沒有檔案，僅保留時間戳記的資料列
        row = {'agree_type':None, 'folder': folder, 'file': None, 'result': None, 'timestamp': timestamp}
        df = df.append(row, ignore_index=True)

print(df)

logger.info(df)
# group by 同一個 folder 名稱下的 result 
grouped_df = df.groupby('folder')
filtered_rows = []


for folder, group in grouped_df:
    if (group[(group['agree_type'] == '1') & (group['result'] == 'agree')].any()).all() and (group[(group['agree_type'] == '2') & (group['result'] == 'agree')].any()).all():
        result = '企業同意/負責人同意且通過'
    elif (group[(group['agree_type'] == '1') & (group['result'] == 'agree')].any()).all() and (group[(group['agree_type'] == '2') & (group['result'] == 'disagree')].any()).all():
        result = '企業同意/負責人不同意且通過'
    elif (group[(group['agree_type'] == '1') & (group['result'] == 'disagree')].any()).all() and (group[(group['agree_type'] == '2') & (group['result'] == 'agree')].any()).all():
        result = '企業不同意/負責人同意且通過'
    elif (group[(group['agree_type'] == '1') & (group['result'] == 'disagree')].any()).all() and (group[(group['agree_type'] == '2') & (group['result'] == 'disagree')].any()).all():
        result = '企業不同意/負責人不同意且通過'
    elif group['result'].notna().any():
        result = 'Fail'
    else:
        result = None



    timestamp = group['timestamp'].iloc[0]  # 使用第一筆資料的 timestamp
    filtered_rows.append({'pno': folder, 'ai_result': result, 'update_dt': timestamp})
   
    
    
# 建立新的 DataFrame
filtered_df = pd.DataFrame(filtered_rows)

filtered_df = filtered_df[['pno', 'ai_result', 'update_dt']].drop_duplicates()

print("result:")
logger.info("result:")
print(filtered_df)
logger.info(filtered_df)




current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d")  # 把":"換成"-"，因為":"不能作為路徑的一部分


# 指定暫存資料夾的路徑
# temp_fail_folder_path = f'datasets\\fail\\{formatted_time}'
# temp_succeed_folder_path = f'datasets\\succeed\\{formatted_time}'
temp_all_agree_folder_path = f'datasets/all_agree/{formatted_time}' #全同意
temp_E_agree_P_disagree_folder_path = f'datasets/E_agree_P_disagree/{formatted_time}' # 企業同意 負責人不同意
temp_E_disagree_P_agree_folder_path = f'datasets/E_disagree_P_agree/{formatted_time}' # 企業不同意 負責人同意
temp_all_disagree_folder_path = f'datasets/all_disagree/{formatted_time}' #全不同意
temp_Fail_folder_path = f'datasets/Fail/{formatted_time}' # 人工審核

for filename in folder_list:
    file_path = os.path.join(folder_path, filename)
    
    try: 
        if len(filtered_df[filtered_df['pno'].str.contains(filename)]) > 0 :

            if filtered_df[filtered_df['pno'].str.contains(filename)]['ai_result'].iloc[0] == '企業同意/負責人同意且通過' :

                # 移動資料夾或刪除檔案
                if os.path.isdir(file_path):
                    os.makedirs(temp_all_agree_folder_path, exist_ok=True)

                    # 移動資料夾至暫存資料夾
                    shutil.move(file_path, temp_all_agree_folder_path)
                elif os.path.isfile(file_path):
                    # 刪除檔案
                    os.remove(file_path)
            elif filtered_df[filtered_df['pno'].str.contains(filename)]['ai_result'].iloc[0] == '企業同意/負責人不同意且通過' :

                # 移動資料夾或刪除檔案
                if os.path.isdir(file_path):
                    os.makedirs(temp_E_agree_P_disagree_folder_path, exist_ok=True)
                    # 移動資料夾至暫存資料夾
                    shutil.move(file_path, temp_E_agree_P_disagree_folder_path)
                elif os.path.isfile(file_path):
                    # 刪除檔案
                    os.remove(file_path)

            elif filtered_df[filtered_df['pno'].str.contains(filename)]['ai_result'].iloc[0] == '企業不同意/負責人同意且通過' :

                # 移動資料夾或刪除檔案
                if os.path.isdir(file_path):
                    os.makedirs(temp_E_disagree_P_agree_folder_path, exist_ok=True)
                    # 移動資料夾至暫存資料夾
                    shutil.move(file_path, temp_E_disagree_P_agree_folder_path)
                elif os.path.isfile(file_path):
                    # 刪除檔案
                    os.remove(file_path)

            elif filtered_df[filtered_df['pno'].str.contains(filename)]['ai_result'].iloc[0] == '企業不同意/負責人不同意且通過' :

                # 移動資料夾或刪除檔案
                if os.path.isdir(file_path):
                    os.makedirs(temp_all_disagree_folder_path, exist_ok=True)
                    # 移動資料夾至暫存資料夾
                    shutil.move(file_path, temp_all_disagree_folder_path)
                elif os.path.isfile(file_path):
                    # 刪除檔案
                    os.remove(file_path)
            elif filtered_df[filtered_df['pno'].str.contains(filename)]['ai_result'].iloc[0] == 'Fail' :

                # 移動資料夾或刪除檔案
                if os.path.isdir(file_path):
                    os.makedirs(temp_Fail_folder_path, exist_ok=True)
                    # 移動資料夾至暫存資料夾
                    shutil.move(file_path, temp_Fail_folder_path)
                elif os.path.isfile(file_path):
                    # 刪除檔案
                    os.remove(file_path)
    except Exception as e:
        print(e)
        logger.error(e)





filtered_df.columns = ['案件編號','ai_result','update_dt']
combine = pd.merge(dfs,filtered_df,on = '案件編號',how = 'outer',indicator = True)


pd.set_option('display.max_columns', None)
combine.sort_values('案件編號')
# combine[['案件編號', '銀行代碼', '分行代碼', '戶名(公司名稱)', '統一編號', '授權方式', '企業電信授權書',
#        '是否同意企業發查', '授權書經辦人', '企業負責人電信授權書', '是否負責人企業發查', '申請時間', 'ai_result', 'update_dt', '_merge']]


print(combine)
logger.info(combine)

# ----------------------- 以下待修改--------------------------------
submit_list_1 = combine[combine['ai_result'] == '企業同意/負責人同意且通過'].reset_index(drop=True)
submit_list_2 = combine[combine['ai_result'] == '企業同意/負責人不同意且通過'].reset_index(drop=True)
submit_list_3 = combine[combine['ai_result'] == '企業不同意/負責人同意且通過'].reset_index(drop=True)
submit_list_4 = combine[combine['ai_result'] == '企業不同意/負責人不同意且通過'].reset_index(drop=True)
print("要送出的清單-企業同意/負責人同意且通過..")
print(submit_list_1)
logger.info("要送出的清單-企業同意/負責人同意且通過..")
logger.info(submit_list_1)

print("要送出的清單-企業同意/負責人不同意且通過..")
print(submit_list_2)
logger.info("要送出的清單-企業同意/負責人不同意且通過..")
logger.info(submit_list_2)

print("要送出的清單-企業不同意/負責人同意且通過..")
print(submit_list_3)
logger.info("要送出的清單-企業不同意/負責人同意且通過..")
logger.info(submit_list_3)

print("要送出的清單-企業不同意/負責人不同意且通過..")
print(submit_list_4)

logger.info("要送出的清單-企業不同意/負責人不同意且通過..")
logger.info(submit_list_4)
print("送出結果..")
logger.info("送出結果..")
import requests
from concurrent.futures import ThreadPoolExecutor



submit_list_1['send_param'] = send_param
submit_list_1['ctxt'] = 1
submit_list_2['send_param'] = send_param
submit_list_2['ctxt'] = 2
submit_list_3['send_param'] = send_param
submit_list_3['ctxt'] = 3
submit_list_4['send_param'] = send_param
submit_list_4['ctxt'] = 4

def submit(row):
    try:
        send_param = row['send_param']
        pno = row['案件編號']
        cnumber = row['統一編號']
        branch = row['分行代碼']
        bankcode = row['銀行代碼']

        ctxt = int(row['ctxt'])

        main_url2 = 'https://0800056476.sme.gov.tw/smeloans/tele_credit/tele_credit_a2_list.php'
        x = main_url2 + f'?send={send_param}&pno={pno}&cnumber={cnumber}&branch={branch}&bankcode={bankcode}&status=Y&ctxt={ctxt}'
        cookies = {
            'PHPSESSID': cookie,
        }
        print(f"pno:{pno}")
        response = requests.get(x, cookies=cookies)
    except Exception as e:
        print(e)
        snd_line(f"送出審核發生錯誤:\n{e}")

# 使用ThreadPoolExecutor提交任务
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(submit, submit_list_1.to_dict('records'))
    executor.map(submit, submit_list_2.to_dict('records'))
    executor.map(submit, submit_list_3.to_dict('records'))
    executor.map(submit, submit_list_4.to_dict('records'))



## line回報

total_num = len(combine)
all_agree_num = len(combine[combine['ai_result'] == '企業同意/負責人同意且通過'])
E_agree_P_disagree_num = len(combine[combine['ai_result'] == '企業同意/負責人不同意且通過'])
E_disagree_P_agree_num = len(combine[combine['ai_result'] == '企業不同意/負責人同意且通過'])
all_disagree_num = len(combine[combine['ai_result'] == '企業不同意/負責人不同意且通過'])
Fail_num = len(combine[combine['ai_result'] == 'Fail'])
wait_to_check_num = len(combine[combine['_merge'] == 'left_only'])

msg = f"\n總案件:{total_num}\n-企業同意/負責人同意且通過案件數:{all_agree_num}\n-企業不同意/負責人同意且通過案件數:{E_disagree_P_agree_num}\n-企業同意/負責人不同意且通過案件數:{E_agree_P_disagree_num}\n-企業不同意/負責人不同意且通過案件數:{all_disagree_num}\n-失敗案件數:{Fail_num}\n-等待人工審核案件數:{wait_to_check_num}"
logger.info(msg)
snd_line(msg)



