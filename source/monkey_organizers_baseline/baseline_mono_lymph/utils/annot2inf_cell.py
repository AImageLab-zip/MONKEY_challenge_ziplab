import xml.etree.ElementTree as ET

def annot2inf_cell(xml_path, output_path):
    # parsing the annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculating the bbox in pixels

    # iterating through the dot annotation.
    for A in root.iter('Annotation'):
        if (A.get('PartOfGroup') == 'lymphocytes') or (A.get('PartOfGroup') == 'monocytes'):
            # print(A.get('PartOfGroup'))
            A.attrib['PartOfGroup'] = 'inf_cell'
            # print(A.attrib['PartOfGroup'])

    # iterating through the dot annotation.
    for A in root.iter('Group'):
        # print(A.attrib['Name'])
        if (A.attrib['Name'] == 'lymphocytes'):
            A.attrib['Name'] = 'inf_cell'
            # print(A.attrib['Name'])
        if (A.attrib['Name'] == 'monocytes'):
            A.attrib['Name'] = 'Empty'
            # print(A.attrib['Name'])

    # writing the new annotation file
    tree.write(output_path)


# # example:
# xml_path = r"./file.xml"
# output_path = r"./file_polygon_infcell.xml"
# annot2inf_cell(xml_path, output_path)
