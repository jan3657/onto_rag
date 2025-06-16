# scripts/reduce_evaluation_set.py

import xml.etree.ElementTree as ET
import os
import sys
from collections import defaultdict

# Add project root to Python path to allow direct imports if needed in the future
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- Configuration ---
# Assumes the data folder is at the project root
INPUT_XML_PATH = os.path.join(PROJECT_ROOT, "data", "CafeteriaFCD_foodon.xml")
OUTPUT_XML_PATH = os.path.join(PROJECT_ROOT, "data", "CafeteriaFCD_foodon_unique.xml")

def create_unique_dataset(input_file: str, output_file: str):
    """
    Parses an XML annotation file and creates a new, smaller XML file
    containing only one instance of each unique (text, semantic_tags) pair.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {input_file}: {e}")
        return

    # A set to store the unique identifiers of annotations we've already added.
    # The identifier will be a tuple: (UPPERCASE_TEXT, sorted_tuple_of_tags)
    seen_annotations = set()
    
    # These will be the root and single document of our new XML file
    new_root = ET.Element("collection")
    new_doc = ET.SubElement(new_root, "document")
    new_doc.set("id", "unique_annotations_document")

    total_annotations_processed = 0
    unique_annotations_found = 0

    # Find all annotation tags anywhere in the document
    for annotation_node in root.findall('.//annotation'):
        total_annotations_processed += 1
        
        text_node = annotation_node.find('text')
        tags_node = annotation_node.find('infon[@key="semantic_tags"]')

        # Ensure both text and tags exist to form a valid entry
        if (text_node is not None and text_node.text and
                tags_node is not None and tags_node.text):
            
            # 1. Normalize the text to be case-insensitive
            text = text_node.text.strip().upper()

            # 2. Normalize the tags to be order-insensitive
            raw_tags = tags_node.text.strip()
            # Split by ';', strip whitespace, filter out any empty strings, and sort
            sorted_tags = sorted([tag.strip() for tag in raw_tags.split(';') if tag.strip()])

            # Create a unique, hashable key for this annotation
            # We convert the list of tags to a tuple to make it hashable for the set
            annotation_key = (text, tuple(sorted_tags))

            # 3. If we haven't seen this unique combination before, add it
            if annotation_key not in seen_annotations:
                seen_annotations.add(annotation_key)
                # Append the original annotation_node to our new document
                new_doc.append(annotation_node)
                unique_annotations_found += 1

    print(f"Processed {total_annotations_processed} total annotations.")
    print(f"Found {unique_annotations_found} unique (text, tags) pairs.")

    # Write the new, smaller XML tree to the output file
    new_tree = ET.ElementTree(new_root)
    # The indent function (Python 3.9+) makes the XML output readable
    if sys.version_info >= (3, 9):
        ET.indent(new_tree)
        
    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Successfully saved unique dataset to: {output_file}")


if __name__ == "__main__":
    create_unique_dataset(INPUT_XML_PATH, OUTPUT_XML_PATH)