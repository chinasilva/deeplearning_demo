import xlrd,xlwt

class Get_data:
    def red_excel(self,path):
        workbook = xlrd.open_workbook(path,encoding_override="utf-8")
        sheets = workbook.sheet_names()
        booksheet = workbook.sheet_by_name(sheets[0])
        col_values = booksheet.col_values(8)
        # booksheet.row_values(0)
        data = list(map(int,col_values))
        return data
get_data =  Get_data()