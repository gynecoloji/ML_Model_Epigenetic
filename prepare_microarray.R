.libPaths("../../../Argha/ChIP_Seq_11022025/Snakemake_ChIP/tmp")
library(stringr)
library(oligo)
library(affy)
library(arrayQualityMetrics)
library(AnnotationDbi)
library(GEOquery)
library(hugene10sttranscriptcluster.db)

gse <- getGEO("GSE118171", GSEMatrix = TRUE)
eset <- gse[[1]]   # usually only one platform
expr <- exprs(eset)
pheno <- pData(eset)
pheno$cl <- str_replace(pheno$title, "Gene expression of ", "")
pheno$cl <- str_replace(pheno$cl, " ", "_")

colnames(expr) <- pheno$cl
expr <- expr[,grep("(SKOV3)|(OVCA429)|(PEO1)|(HEY_)", colnames(expr))]

annot <- select(
  hugene10sttranscriptcluster.db,
  keys = rownames(expr),
  keytype = "PROBEID",
  columns = c(
    "SYMBOL",
    "ENSEMBL"
  )
)

annot <- annot[!duplicated(annot$PROBEID), ]
rownames(annot) <- annot$PROBEID
identical(rownames(annot), rownames(expr))

valid <- !is.na(annot$ENSEMBL)
expr_clean <- expr[valid, ]
rownames(expr_clean) <- annot$ENSEMBL[valid]
colnames(expr_clean)[grep("HEY", colnames(expr_clean))] <- c("HeyA8_rep1", "HeyA8_rep2")

write.csv(
  expr_clean,
  "data/expression/gene_expression.csv",
  row.names = TRUE
) 

cell_line <- str_replace(colnames(expr_clean), "_rep[0-9]+$", "")
expr_merged <- sapply(unique(cell_line), function(cl) {
  rowMeans(expr_clean[, cell_line == cl, drop = FALSE])
})
write.csv(
  expr_merged,
  "data/expression/gene_expression_merged_mean.csv",
  row.names = TRUE
) 
