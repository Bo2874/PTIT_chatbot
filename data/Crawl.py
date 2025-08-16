import os
import json
import requests
from bs4 import BeautifulSoup

METADATA_FILE = "metadata.json"

def crawl_lich_su_truyen_thong():
    url = 'https://ptit.edu.vn/gioi-thieu/tong-quan-hoc-vien/lich-su-truyen-thong'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body = soup.find('div', class_="elementor elementor-27825")

        content_paragraphs = body.find_all(['p', 'h2'])

        table_data = []
        table = body.find('table')
        if table:
            rows = table.find_all('tr')
            for row_id, row in enumerate(rows):
                columns = row.find_all('td')
                if columns:
                    ngay = columns[0].text.strip()
                    noi_dung = columns[1].text.strip()
                    table_data.append(f"Ngày: {ngay}\n{noi_dung}")

        content = "\n".join([p.get_text() for p in content_paragraphs])
        content += "\n" + "\n".join(table_data)

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')


def crawl_y_nghia_logo():
    url = 'https://ptit.edu.vn/gioi-thieu/tong-quan-hoc-vien/y-nghia-logo-hoc-vien'
    response = requests.get(url)
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="elementor-element elementor-element-378168e elementor-widget elementor-widget-text-editor")

        content_paragraphs = body.find_all(['div'])

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in content_paragraphs])
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_tam_nhin_su_mang():
    url = 'https://ptit.edu.vn/gioi-thieu/tong-quan-hoc-vien/tam-nhin-su-mang'
    response = requests.get(url)
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="elementor-element elementor-element-9c44e2e elementor-widget elementor-widget-text-editor")

        content_paragraphs = body.find_all(['div'])

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in content_paragraphs])
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_triet_ly_giao_duc():
    url = 'https://ptit.edu.vn/gioi-thieu/tong-quan-hoc-vien/triet-ly-giao-duc'
    response = requests.get(url)
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="e-con-inner")
        content_paragraphs = body.find_all(['h2', 'p'])

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in content_paragraphs])
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_chien_luoc_phat_trien():
    url = 'https://ptit.edu.vn/gioi-thieu/tong-quan-hoc-vien/chien-luoc-phat-trien-giai-doan-2021-2025-tam-nhin-2030'
    response = requests.get(url)
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="e-con-inner")
        content_paragraphs = body.find_all(['h2', 'h4', 'p'])

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in content_paragraphs])
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_dang_uy_hoc_vien():
    # URL của trang cần crawl
    url = 'https://ptit.edu.vn/gioi-thieu/co-cau-to-chuc/dang-uy-hoc-vien'

    # Gửi yêu cầu đến trang web
    response = requests.get(url)

    # Kiểm tra xem yêu cầu có thành công không
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="the_content_wrapper")

        content_paragraphs = body.find_all(['h5'])
        # tìm bảng
        table_data = []
        table = body.find('table', id="tblNoidung")
        rows = table.find_all('tr')
        for row_id, row in enumerate(rows):
            columns = row.find_all('td')
            if columns:
                name = columns[1].text.strip()
                position = columns[2].text.strip()
                table_data.append(f"{name} là {position}")

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in content_paragraphs])
        content += "\n".join(table_data)
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_hoi_dong_hoc_vien():
    # URL của trang cần crawl
    url = 'https://ptit.edu.vn/gioi-thieu/co-cau-to-chuc/hoi-dong-hoc-vien'

    # Gửi yêu cầu đến trang web
    response = requests.get(url)

    # Kiểm tra xem yêu cầu có thành công không
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="the_content_wrapper")

        title_1 = body.find_all(['h4'])
        # tìm bảng
        table_data = []
        table = body.find('table')
        rows = table.find_all('tr')
        for row_id, row in enumerate(rows):
            columns = row.find_all('td')
            if columns:
                name = columns[1].text.strip()
                position = columns[2].text.strip()
                table_data.append(f"{name}, {position}")
        # tìm text
        content_paragraphs = body.find_all(['p'])
        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in title_1]) + "\n\n"
        content += "\n".join(table_data) + "\n\n"
        content += "\n".join([p.get_text() for p in content_paragraphs])
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_ban_giam_doc_hoc_vien():
    url = 'https://ptit.edu.vn/gioi-thieu/co-cau-to-chuc/ban-giam-doc-hoc-vien'
    response = requests.get(url)
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="e-con-inner")
        content_paragraphs = body.find_all(['h1', 'h2', 'p'])

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join([p.get_text() for p in content_paragraphs])
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_hoi_dong_khoa_hoc_va_dao_tao():
    # URL của trang cần crawl
    url = 'https://ptit.edu.vn/gioi-thieu/co-cau-to-chuc/hoi-dong-khoa-hoc-va-dao-tao'

    # Gửi yêu cầu đến trang web
    response = requests.get(url)

    # Kiểm tra xem yêu cầu có thành công không
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="the_content_wrapper")

        # tìm bảng
        table_data = []
        table = body.find('table')
        rows = table.find_all('tr')
        for row_id, row in enumerate(rows):
            columns = row.find_all('td')
            if columns:
                name = columns[1].text.strip()
                position = columns[2].text.strip()
                table_data.append(f"{name}, {position}")

        # Kết hợp nội dung vào một chuỗi
        content = "\n".join(table_data)
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_hoi_dong_giao_su_co_so():
    # URL của trang cần crawl
    url = 'https://ptit.edu.vn/gioi-thieu/co-cau-to-chuc/hoi-dong-giao-su-co-so'

    # Gửi yêu cầu đến trang web
    response = requests.get(url)

    # Kiểm tra xem yêu cầu có thành công không
    if response.status_code == 200:
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Tìm tiêu đề của trang
        title = soup.title.string
        # truy cập vào phần body
        body = soup.find('div', class_="the_content_wrapper")

        # tìm bảng
        table_contents = []
        tables = body.find_all('table')
        for table in tables:
            table_data = []
            rows = table.find_all('tr')
            for row_id, row in enumerate(rows):
                columns = row.find_all('td')
                row_data = "";
                if columns:
                    for i in range(len(columns)):
                        row_data += columns[i].text.strip()
                        if i != len(columns) - 1:
                            row_data += ", "  
                table_data.append(row_data) 
            table_content = "\n".join(table_data)
            table_contents.append(table_content)

        # Kết hợp nội dung vào một chuỗi
        content = "\n\n".join(table_contents)
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Lỗi: Không thể truy cập trang. Mã trạng thái: {response.status_code}')

def crawl_nguon_nhan_luc():
    url = 'https://ptit.edu.vn/gioi-thieu/nguon-nhan-luc'
    # Gui y/c den trang web
    response = requests.get(url)
    # Kiem tra xem y/c co thanh cong khong
    if response.status_code == 200:
        # Phan tich HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tieu de
        title = soup.title.string
        # Truy cap vao body
        body = soup.find('div', class_ = 'e-con-inner')
        content_body = body.find_all(['span'])
        # Ket hop noi dung vao chuoi
        content = "\n".join([p.get_text() for p in content_body]) # Xuong dong o moi lan het 1 doan van ban
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Loi: Khong the truy cap trang. Ma trang thai: {response.status_code}')    

def crawl_co_so_vat_chat():
    url = 'https://ptit.edu.vn/gioi-thieu/co-so-vat-chat'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body = soup.find('div', class_='e-con-inner')
        content_body = body.find_all(['p'])
        content = "\n".join(p.get_text() for p in content_body)
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Loi: Khong the truy cap trang. Ma trang thai:{response.status_code}')

def crawl_khoa_hoc_may_tinh():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-khoa-hoc-may-tinh/#tai_lieu_dao_tao"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_cong_nghe_thong_tin():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-thong-tin/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[2].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-2-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))
        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_an_toan_thong_tin():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-an-toan-thong-tin/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))
        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_ky_thuat_dien_tu_vien_thong():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ky-thuat-dien-tu-vien-thong/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[2].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-2-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_cong_nghe_ky_thuat_dien_dien_tu():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-ky-thuat-dien-dien-tu/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[2].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-2-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[3].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-3-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_cong_nghe_da_phuong_tien():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-da-phuong-tien/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_truyen_thong_da_phuong_tien():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-truyen-thong-da-phuong-tien/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text()

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_marketing():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-marketing/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[2].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-2-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_thuong_mai_dien_tu():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-thuong-mai-dien-tu/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_quan_tri_kinh_doanh():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-quan-tri-kinh-doanh/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[2].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-2-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_ke_toan():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ke-toan/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_cong_nghe_thong_tin_he_clc():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-thong-tin-he-clc/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[2].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-2-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"


        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_ky_thuat_dieu_khien_va_tu_dong_hoa():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ky-thuat-dieu-khien-va-tu-dong-hoa/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        chuan_dau_ra += li_list[1].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_ke_toan_2022():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ke-toan-2022/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"

        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_bao_chi_journalism():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-bao-chi-journalism/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_cong_nghe_tai_chinh_fintech():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-tai-chinh-fintech/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_ky_thuat_du_lieu_nganh_mang_may_tinh_va_truyen_thong_du_lieu():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-ky-thuat-du-lieu-nganh-mang-may-tinh-va-truyen-thong-du-lieu/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_marketing_he_clc():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-marketing-he-clc/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    
def crawl_cong_nghe_internet_van_vat_iot():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-internet-van-vat-iot/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_ke_toan_chat_luong_cao_chuan_quoc_te_acca():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ke-toan-chat-luong-cao-chuan-quoc-te-acca/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_chuong_trinh_quan_he_cong_chung_nganh_marketing():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-quan-he-cong-chung-nganh-marketing/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_chuong_trinh_thiet_ke_va_phat_trien_game():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-thiet-ke-va-phat-trien-game/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,8):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_chuong_trinh_cong_nghe_thong_tin_viet_nhat():
    url = "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-cong-nghe-thong-tin-viet-nhat/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        body1 = soup.find('ul', class_='column_4 mtop')
        card = body1.find_all(['label'])
        card2 = body1.find_all(['strong'])
        content_pair = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(card, card2)]
        content1 = "\n".join(content_pair)

        body2 = soup.find('div', class_='ova_dir_content')
        body2_content = body2.find_all(['section'])
        tong_quan = body2_content[0].get_text()
        chuan_dau_ra = body2_content[1].get_text() + "\n"

        li_list = body2_content[2].find('ul', class_='nav-tab').find_all('li')
        chuan_dau_ra += li_list[0].get_text().strip()
        chuan_dau_ra += "\n"
        for i in range(0,9):
            hoc_ky = body2_content[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            chuan_dau_ra += label.get_text().strip()
            chuan_dau_ra += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            chuan_dau_ra += content_pair
            chuan_dau_ra += "\n\n"
        
        nghe_nghiep = body2_content[3].get_text()
        hoc_phi = body2_content[4].get_text()
        dk_tuyen_sinh = body2_content[5].get_text()
        qt_nhap_hoc = body2_content[6]
        content_qtnh = ''
        content_qtnh += qt_nhap_hoc.find_all(['h3'])[0].get_text() + "\n"
        content_qtnh += "\n".join(p.get_text() for p in qt_nhap_hoc.find_all(['label']))

        content = content1 + tong_quan + chuan_dau_ra + nghe_nghiep + hoc_phi + dk_tuyen_sinh + content_qtnh
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_chtrinh_cntt_dinh_huong_ung_dung():
    url = 'https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-cong-nghe-thong-tin-dinh-huong-ung-dung/'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        
        body1 = soup.find('ul', class_='column_4 mtop')
        labels = body1.find_all('label')
        strongs = body1.find_all('strong')
        content_pairs = [f"{label.get_text()}: {strong.get_text()}" for label, strong in zip(labels, strongs)]
        content1 = "\n".join(content_pairs)

        body2 = soup.find('div', class_='ova_dir_content')
        content2 = body2.find_all(['section'])
        content1 += content2[0].get_text()
        content1 += content2[1].get_text()
        content1 += "\n"
        
        li_list = content2[2].find('ul', class_='nav-tab').find_all('li')
        content1 += li_list[0].get_text().strip()
        content1 += "\n"
        for  i in range(0, 8):
            hoc_ky = content2[2].find('div', id=f"tab-0-{i}")
            label = hoc_ky.find('label')
            content1 += label.get_text()
            content1 += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            content1 += content_pair
            content1 += "\n\n"
        content1 += li_list[1].get_text().strip()
        content1 += "\n"
        for  i in range(0, 8):
            hoc_ky = content2[2].find('div', id=f"tab-1-{i}")
            label = hoc_ky.find('label')
            content1 += label.get_text()
            content1 += "\n"
            so_tin = hoc_ky.find_all('div', class_='tag')
            mon_hoc = hoc_ky.find_all('div', class_='title')
            pairs = [f"{mon.get_text().strip()}: {tin.get_text().strip()}" for mon, tin in zip(mon_hoc, so_tin)]
            content_pair = "\n".join(pairs)
            content1 += content_pair
            content1 += "\n\n" 

        content1 += content2[3].get_text()
        content1 += content2[4].get_text()
        content1 += content2[5].get_text()
        
        this_labels = content2[6].find_all(['h3', 'label'])
        content1 += "\n"
        content1 += "\n".join(p.get_text() for p in this_labels)

        content = content1 
        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]
    else:
        print(f'Loi khong the truy cap trang. Ma trang thai: {response.status_code}')

def crawl_doi_ngu_can_bo_phong_dao_tao():
    url = "https://daotao.ptit.edu.vn/gioi-thieu/doi-ngu-can-bo-phong-dao-tao/?fbclid=IwY2xjawFqPQtleHRuA2FlbQIxMAABHd7TdZvocI4f55HFKcS05Bx3W675W5chbcksu32xuYUg1wBA0zxevlsUpA_aem_PWX43vwiVdCocf2reGQHCg"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string
        
        body = soup.find('div', class_="e-con-inner")
        
        content_paragraphs = body.find_all(['h2', 'p'])
        content = "\n".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_co_so_vat_chat_hoc_vien():
    url = "https://daotao.ptit.edu.vn/gioi-thieu/co-so-vat-chat-hoc-vien/?fbclid=IwY2xjawFqPQ5leHRuA2FlbQIxMAABHUQCgNBz3jkYC7y-FQqI4oPMgb2RJU5IXluA_CEE7caNQfXsxfrd40blLQ_aem__-7CSXC3fgcHU884IeWj8Q"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="elementor elementor-312")
       
        content_paragraphs = body.find_all(['h2', 'p'])
        content = "\n".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_chuc_nang_nhiem_vu():
    url = "https://daotao.ptit.edu.vn/gioi-thieu/chuc-nang-nhiem-vu/?fbclid=IwY2xjawFqPQhleHRuA2FlbQIxMAABHWL3Dm0ObGrB_RQN9J9y4YSNyEeanOwukn8slsCWAbfdfXpflZcYuIGI7Q_aem_3aMZLa07sTjMxCXIGJdvFg"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="elementor elementor-286")
       
        content_paragraphs = body.find_all(['h5', 'p'])
        content = " ".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_cau_lac_bo_sinh_vien():
    url = "https://ptit.edu.vn/sinh-vien/cau-lac-bo-sinh-vien"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="e-con-inner")
       
        content_paragraphs = body.find_all(['h3', 'p'])
        content = " ".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_doan_thanh_nien():
    url = "https://ptit.edu.vn/sinh-vien/doan-thanh-nien"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="e-con-inner")
       
        content_paragraphs = body.find_all(['h3', 'p'])
        content = " ".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_khai_truong_phong_thuc_hanh_cong_nghe_game():
    url = "https://ptit.edu.vn/tin-tuc/hoc-vien-cong-nghe-buu-chinh-vien-thong-khai-truong-phong-thuc-hanh-cong-nghe-game"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="ova_dir_content")
       
        content_paragraphs = body.find_all(['h1', 'p'])
        content = " ".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_dai_su_NVIDIA():
    url = "https://ptit.edu.vn/tin-tuc/giang-vien-hoc-vien-cong-nghe-buu-chinh-vien-thong-duoc-cong-nhan-la-dai-su-nvidia"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="ova_dir_content")
       
        content_paragraphs = body.find_all(['h1', 'p'])
        content = " ".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]

def crawl_thanh_lap_khoa_AI():
    url = "https://ptit.edu.vn/tin-tuc/truong-dai-hoc-dau-tien-tai-viet-nam-lap-khoa-tri-tue-nhan-tao"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string

        body = soup.find('div', class_="ova_dir_content")
       
        content_paragraphs = body.find_all(['h1', 'p'])
        content = " ".join([p.get_text() for p in content_paragraphs])

        data = dict(title=title, url=url, content=content)

        # Đọc metadata cũ
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Thêm dữ liệu mới
        metadata.append(data)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return [data]



if __name__ == '__main__':
    # crawl_lich_su_truyen_thong()
    # crawl_y_nghia_logo()
    # crawl_tam_nhin_su_mang()
    # crawl_triet_ly_giao_duc()
    # crawl_chien_luoc_phat_trien()
    # crawl_dang_uy_hoc_vien()
    # crawl_hoi_dong_hoc_vien()
    # crawl_ban_giam_doc_hoc_vien()
    # crawl_hoi_dong_khoa_hoc_va_dao_tao()
    # crawl_hoi_dong_giao_su_co_so()
    # crawl_nguon_nhan_luc()  
    # crawl_co_so_vat_chat()
    # crawl_khoa_hoc_may_tinh()
    # crawl_cong_nghe_thong_tin()
    # crawl_an_toan_thong_tin()
    # crawl_ky_thuat_dien_tu_vien_thong()
    # crawl_cong_nghe_ky_thuat_dien_dien_tu()
    # crawl_cong_nghe_da_phuong_tien()
    # crawl_truyen_thong_da_phuong_tien()
    # crawl_marketing()
    # crawl_thuong_mai_dien_tu()
    # crawl_quan_tri_kinh_doanh()
    # crawl_ke_toan()
    # crawl_cong_nghe_thong_tin_he_clc()
    # crawl_ky_thuat_dieu_khien_va_tu_dong_hoa()
    # crawl_ke_toan_2022()
    # crawl_bao_chi_journalism()
    # crawl_cong_nghe_tai_chinh_fintech()
    # crawl_ky_thuat_du_lieu_nganh_mang_may_tinh_va_truyen_thong_du_lieu()
    # crawl_marketing_he_clc()
    # crawl_cong_nghe_internet_van_vat_iot()
    # crawl_ke_toan_chat_luong_cao_chuan_quoc_te_acca()
    # crawl_chuong_trinh_quan_he_cong_chung_nganh_marketing()
    # crawl_chuong_trinh_thiet_ke_va_phat_trien_game()
    # crawl_chuong_trinh_cong_nghe_thong_tin_viet_nhat()
    # crawl_chtrinh_cntt_dinh_huong_ung_dung()
    # crawl_doi_ngu_can_bo_phong_dao_tao()
    # crawl_co_so_vat_chat_hoc_vien()
    # crawl_chuc_nang_nhiem_vu()
    # crawl_cau_lac_bo_sinh_vien()
    # crawl_doan_thanh_nien()
    # crawl_khai_truong_phong_thuc_hanh_cong_nghe_game()
    # crawl_dai_su_NVIDIA()
    # crawl_thanh_lap_khoa_AI()