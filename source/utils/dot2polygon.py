import xml.etree.ElementTree as ET

def dot2polygon(xml_path, lymphocyte_half_box_size, monocytes_half_box_size, min_spacing, output_path):
    '''
    Convert dot annotations (single points) into polygon annotations (squares) for lymphocytes and monocytes.
    
    Parameters
    ----------
    xml_path : str
        The path of the annotation file, for example: "root/sub_root/filename.xml"
        
    lymphocyte_half_box_size : float
        The half-size of the square bounding box around a lymphocyte dot in micrometers (µm).
        For instance, 4.5 µm is commonly used for lymphocytes.
        
    monocytes_half_box_size : float
        The half-size of the square bounding box around a monocyte dot in micrometers (µm).
        For example, 11.0 µm is often used for monocytes.
        
    min_spacing : float
        The micrometer-to-pixel ratio (µm/px).
        This value is used to convert the specified bounding box sizes (in µm) to pixels.
        For example, if min_spacing = 0.5, it means 1 pixel = 0.5 µm, so a 4.5 µm half-box
        translates to 4.5 / 0.5 = 9 pixels.
        
    output_path : str
        The file path where the updated XML with polygon annotations should be saved.
    
    Returns
    -------
    None
        The function modifies the annotation XML and writes it to output_path.
    '''

    # Parse the annotation XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Convert half-box sizes from micrometers (µm) to pixels.
    # Dividing by min_spacing (µm/px) gives the size in pixels.
    lymphocyte_half_box_size = lymphocyte_half_box_size / min_spacing
    monocytes_half_box_size = monocytes_half_box_size / min_spacing

    # Iterate over each Annotation element in the XML.
    for A in root.iter('Annotation'):
        
        # For Lymphocytes:
        # Check if the current annotation is a Dot representing a lymphocyte.
        if (A.get('PartOfGroup') == "lymphocytes") and (A.get('Type') == "Dot"):
            # Change annotation type from Dot to Polygon.
            A.attrib['Type'] = "Polygon"

            # 'child' generally corresponds to a Coordinates element.
            for child in A:
                # Each sub_child is a coordinate element (X, Y).
                for sub_child in child:
                    # Extract original pixel positions of the dot
                    x_value = sub_child.attrib['X']
                    y_value = sub_child.attrib['Y']

                    # Move the original dot coordinate up-left by half_box_size
                    # This sets the top-left corner of the polygon.
                    sub_child.attrib['X'] = str(float(x_value) - lymphocyte_half_box_size)
                    sub_child.attrib['Y'] = str(float(y_value) - lymphocyte_half_box_size)

                # After adjusting the original point, append three more points
                # to form a square polygon around the original dot location.
                #
                # The polygon is defined as a square centered on the original dot:
                #   Top-left:     (X - half_box, Y - half_box) <- already set above
                #   Bottom-left:  (X - half_box, Y + half_box)
                #   Bottom-right: (X + half_box, Y + half_box)
                #   Top-right:    (X + half_box, Y - half_box)
                #
                # Note: We already modified the first point above. Now we add the other three.
                
                child.append(ET.Element(
                    sub_child.tag,
                    Order='1',
                    X=str(float(x_value) - lymphocyte_half_box_size),
                    Y=str(float(y_value) + lymphocyte_half_box_size)
                ))
                child.append(ET.Element(
                    sub_child.tag,
                    Order='2',
                    X=str(float(x_value) + lymphocyte_half_box_size),
                    Y=str(float(y_value) + lymphocyte_half_box_size)
                ))
                child.append(ET.Element(
                    sub_child.tag,
                    Order='3',
                    X=str(float(x_value) + lymphocyte_half_box_size),
                    Y=str(float(y_value) - lymphocyte_half_box_size)
                ))

        # For Monocytes:
        # Similar process as above, but using monocytes_half_box_size.
        if (A.get('PartOfGroup') == "monocytes") and (A.get('Type') == "Dot"):
            # Change annotation type from Dot to Polygon.
            A.attrib['Type'] = "Polygon"

            for child in A:
                for sub_child in child:
                    x_value = sub_child.attrib['X']
                    y_value = sub_child.attrib['Y']

                    # Adjust the original coordinate to set the top-left corner.
                    sub_child.attrib['X'] = str(float(x_value) - monocytes_half_box_size)
                    sub_child.attrib['Y'] = str(float(y_value) - monocytes_half_box_size)

                # Append the other three corners of the square.
                child.append(ET.Element(
                    sub_child.tag,
                    Order='1',
                    X=str(float(x_value) - monocytes_half_box_size),
                    Y=str(float(y_value) + monocytes_half_box_size)
                ))
                child.append(ET.Element(
                    sub_child.tag,
                    Order='2',
                    X=str(float(x_value) + monocytes_half_box_size),
                    Y=str(float(y_value) + monocytes_half_box_size)
                ))
                child.append(ET.Element(
                    sub_child.tag,
                    Order='3',
                    X=str(float(x_value) + monocytes_half_box_size),
                    Y=str(float(y_value) - monocytes_half_box_size)
                ))

    # Write the modified tree (with polygons replacing dots) to the output file.
    tree.write(output_path)
