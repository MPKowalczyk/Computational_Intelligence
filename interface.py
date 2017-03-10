# Input interface for computational intelligence

from xlrd import open_workbook

def interface(name):
    xls=open_workbook(name)
    sheet=xls.sheet_by_index(0)
    cols=sheet.row_len(0)
    matrix=[list() for x in range(cols)]
    for x in range(0,cols):
        matrix[x]=sheet.col(x)
    return matrix